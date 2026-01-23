#
# 6. GRAPH NODES
#

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
# LangChain / LangGraph Core
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from legal_modules.node_helpers import *
from legal_modules.setup import db, embeddings, llm
from legal_modules.state import AgentState
from legal_modules.tools import web_search_tool, websearch_llm
from legal_modules.utils import (get_user_doc_collection_name,
                                 retrieve_filtered_documents)

from langgraph_legal_ai.legal_modules.prompts import *


#  1. Ingestion
def ingest_document_if_needed(state: AgentState) -> dict:
    """
    Ingest document if needed.

    Checks if a document path is provided and if so, loads the document and chunks it into a ChromaDB collection.
    If the document text already exists in the state, it skips the ingestion step.

    Returns a dictionary with the following keys:
    - user_doc_collection: the name of the ChromaDB collection containing the ingested document
    - current_step: the name of the current step in the workflow
    - review_count: the number of times the document has been reviewed
    """

    NODE_NAME = "ingest_document_if_needed"
    print(f"Node: {NODE_NAME}")

    document_path = state.get("document_path")
    if not document_path:
        return {
            "user_doc_collection": None,
            "current_step": "ingest_document_if_needed",
            "review_count": 0,
        }

    # Check to avoid re-processing if text exist
    if state.get("document_text"):
        return {"current_step": "ingest_document_if_needed", "review_count": 0}

    try:
        loader = PyMuPDFLoader(document_path)
        docs = loader.load()
        document_text = "".join(doc.page_content for doc in docs)
    except Exception as e:
        return {
            "error": f"PDF parsing failed: {str(e)}",
            "current_step": "ingest_document_if_needed",
        }

    # Chunking & Embedding
    collection_name = chunk_and_save_to_chromadb(document_text, document_path)

    return {
        "document_text": document_text,
        "user_doc_collection": collection_name,
        "current_step": "ingest_document_if_needed",
        "review_count": 0,
    }


#  2. Decomposition
def decompose_to_analysis_units(state: AgentState) -> dict:
    """
    Decompose the input query to analysis units.
    - Classify Intent.
    - Generate Analysis Units.
    - Generate Actions.
    - Enhance the user Query

    Parameters:
    state (AgentState): The current state of the agent.

    Returns:
    dict: A dictionary containing the analysis units, the intent classification classification results, and the actions needed.
    """

    NODE_NAME = "decompose_to_analysis_units"
    print(f"Node: {NODE_NAME}")

    input_query = state.get("input_query", "")
    document_text = state.get("document_text", "")
    has_document = bool(document_text and document_text.strip())
    chats = state.get("messages", [])
    result = {}

    # Optimise the query, classify intent, and generate actions needed to simplify further process
    try:
        chain = decompose_to_analysis_units_prompt | llm | JsonOutputParser()
        result = chain.invoke(
            {
                "input_query": input_query,
                "chats": chats,
                "has_document": "Yes" if has_document else "No",
                "document_text_stripped": (document_text or "")[:500],
            }
        )

        if not result.get("query_related_to_legal_context", True):
            return {
                "user_query": input_query,
                "intent_classification": result,
                "final_response": "Query is not Related to Legal Context. Please ask Legal Questions.",
                "current_step": "decompose_to_analysis_units",
            }

        intent = result.get("intent", "general")
    except Exception as e:
        print(f"Intent classification failed: {e}")
        intent = "general" if not has_document else "document_general"

    # Generate Analysis Units
    analysis_units = get_analysis_units(
        result, intent, document_text, state.get("user_doc_collection")
    )
    user_query = result.get("optimised_query")
    actions_needed = result.get("actions_needed", [])

    return {
        "user_query": user_query,
        "analysis_units": analysis_units,
        "intent_classification": result if "result" in locals() else {"intent": intent},
        "current_step": "decompose_to_analysis_units",
        "actions_needed": actions_needed,
    }


#  3. Retriever
def retriever(state: AgentState) -> dict:
    """
    Retrieves relevant documents from the database based on the user query and analysis units.

    Parameters:
    state (AgentState): The current state of the agent.

    Returns:
    dict: A dictionary containing the retrieved documents, the current step, and the status of different nodes.
    """
    NODE_NAME = "retriever"
    print(f"Node: {NODE_NAME}")

    # Reterive the relevent documents from the Chroma DB to optimise the answer
    query = state["user_query"]
    analysis_units = state.get("analysis_units", [])
    unique_docs = get_relevant_docs(query, analysis_units, db)

    return {
        "retrieved_docs": unique_docs,
        "current_step": "retriever",
        "doctrinal_done": False,
        "precedent_done": False,
        "remediation_done": False,
        "parallel_join_complete": False,
    }


#  4. Compliance & Loophole Validator
def compliance_and_loophole_validator(state: AgentState) -> dict:
    """
    Validates if the user query is compliant with relevant laws and regulations.

    Parameters:
    state (AgentState): The current state of the agent.

    Returns:
    dict: A dictionary containing the compliance analysis, loophole analysis, and the status of different nodes.
    """

    NODE_NAME = "compliance_and_loophole_validator"
    print(f"Node: {NODE_NAME}")

    # Block Execution if avoided by the decompose_to_analysis_units node
    if is_node_blocked(state, NODE_NAME):
        print("NODE compliance_and_loophole_validator → blocked")
        return {
            "doctrinal_analysis": {
                "summary": "Avoided as it is not needed for user query.",
                "overall_status": "Avoided",
            },
            "loophole_analysis": {
                "summary": "Avoided as it is not needed for user query."
            },
            "doctrinal_done": True,
        }

    analysis_units = state.get("analysis_units", [])
    retrieved_docs = state.get("retrieved_docs", [])
    user_query = state.get("user_query", "")

    if not analysis_units or not retrieved_docs:
        return {
            "doctrinal_analysis": {
                "summary": "Insufficient data.",
                "overall_status": "incomplete",
            },
            "loophole_analysis": {"summary": "No data."},
            "doctrinal_done": True,
        }

    legal_context = "\n\n".join(
        [
            f"[{d.metadata.get('section', 'N/A')}] {d.page_content.strip()}"
            for d in retrieved_docs
        ]
    )

    # Validate if the user query is compliant with relevant laws and regulations and find the loopholes
    try:
        chain = complaince_and_loophole_validator_prompt | llm | JsonOutputParser()
        result = chain.invoke(
            {
                "user_query": user_query,
                "legal_context": legal_context,
                "analysis_units_text": "\n".join(
                    [f"{i+1}. {u}" for i, u in enumerate(analysis_units)]
                ),
            }
        )

        findings = result.get("findings", [])
        loopholes = extract_loopholes(findings)

        return {
            "doctrinal_analysis": {
                "summary": result.get("doctrinal_summary"),
                "findings": findings,
                "overall_status": (
                    "non_compliant"
                    if any(f["status"] != "compliant" for f in findings)
                    else "compliant"
                ),
            },
            "loophole_analysis": {
                "loopholes": loopholes,
                "summary": result.get("loophole_summary"),
            },
            "doctrinal_done": True,
        }
    except Exception as e:
        print(f"Error in compliance node: {e}")
        return {"doctrinal_done": True}


#  5. Precedent Matcher
def precedent_matcher(state: AgentState) -> dict:
    """
    Finds relevant precedents for the user query from the database of legal cases.
    If needed external resources ( web search ) are used to find relevant precedents.

    Parameters:
    state (AgentState): The current state of the agent.

    Returns:
    dict: A dictionary containing the relevant precedents, the current step, and the status of different nodes.
    """

    NODE_NAME = "precedent_matcher"
    print(f"Node: {NODE_NAME}")

    # Block Execution in avoided by the decompose_to_analysis units node
    if is_node_blocked(state, NODE_NAME):
        print("NODE precedent_matcher → blocked")
        return {"precedent_matches": [], "precedent_done": True}

    user_query = state.get("user_query", "")
    messages = state.get("messages", [])

    # Retrieve relevant precedents from the vector database if available
    local_case_docs = [
        d for d in state.get("retrieved_docs", []) if d.metadata.get("case_name")
    ]

    local_case_context = ""
    if local_case_docs:
        local_case_context = "\n\n".join(
            [
                f"Case: {d.metadata['case_name']}\n{d.page_content.strip()}"
                for d in local_case_docs
            ]
        )
    else:
        local_case_context = "No local cases found in the vector database."

    # Use web search if no relevant precedents are found and Find the precedents related to the user query
    return match_precedent(user_query, messages, local_case_context)


#  6. Risk & Remediation Assessor
def risk_and_remediation_assessor(state: AgentState) -> dict:
    """
    Assesses the risk associated with the user query and provides remediation suggestions to mitigate the risk.

    Parameters:
    state (AgentState): The current state of the agent.

    Returns:
    dict: A dictionary containing the risk assessment, remediation suggestions, and the status of different nodes.
    """

    NODE_NAME = "risk_and_remediation_assessor"
    print(f"Node: {NODE_NAME}")

    # Block Execution in avoided by the decompose_to_analysis units node
    if is_node_blocked(state, NODE_NAME):
        print("NODE risk_and_remediation_assessor → blocked")
        return {
            "risk_assessment": {},
            "remediation_suggestions": [],
            "remediation_done": True,
        }

    doctrinal = state.get("doctrinal_analysis", {})
    loopholes = state.get("loophole_analysis", {}).get("loopholes", [])

    # Check if there are any issues
    issues = []
    if doctrinal.get("findings"):
        issues.append(
            f"Doctrinal issues found: {len([f for f in doctrinal['findings'] if f['status']!='compliant'])}"
        )
    if loopholes:
        issues.append(f"Loopholes detected: {len(loopholes)}")

    if not issues:
        return {
            "risk_assessment": {
                "overall_risk": "low",
                "score": 1,
                "rationale": "No issues",
            },
            "remediation_suggestions": [],
            "remediation_done": True,
        }

    # Assess the risks and Generate remediation suggestions
    chain = risk_and_remediation_assessor_prompt | llm | JsonOutputParser()

    try:
        result = chain.invoke({"issues": "\n".join(issues)})
        result["remediation_done"] = True
        return result
    except Exception as e:
        print(f"Error in risk node: {e}")
        return {"remediation_done": True}


#  7. Parallel Join Gate
def parallel_join_gate(state: AgentState) -> dict:
    """
    A node that checks if all the parallel nodes (doctrinal, precedent, and remediation) are complete.
    If all the nodes are complete, it sets `parallel_join_complete` to True.

    Parameters:
    state (AgentState): The current state of the agent.

    Returns:
    dict: A dictionary containing `parallel_join_complete` and the status of different nodes.
    """

    NODE_NAME = "parallel_join_gate"
    print(f"Node: {NODE_NAME}")

    # If all the parallel nodes are complete then set `parallel_join_complete` to True
    if all(
        [
            state.get("doctrinal_done"),
            state.get("precedent_done"),
            state.get("remediation_done"),
        ]
    ):
        if not state.get("parallel_join_complete"):
            return {"parallel_join_complete": True}
    return {}


#  8. Synthesize Verdict
def synthesize_verdict(state: AgentState) -> dict:
    """
    A node that synthesizes a verdict based on the user query, doctrinal analysis, risk assessment, precedent matches, remediation suggestions, and previous chat messages.

    Parameters:
    state (AgentState): The current state of the agent.

    Returns:
    dict: A dictionary containing the synthesized verdict and the current step.
    """

    NODE_NAME = "synthesize_verdict"
    print(f"Node: {NODE_NAME}")

    user_query = state["user_query"]
    analysis_units = state.get("analysis_units", [])
    doctrinal = state.get("doctrinal_analysis", {})
    risk = state.get("risk_assessment", {})
    precedents = state.get("precedent_matches", [])
    remediation = state.get("remediation_suggestions", [])
    messages = state.get("messages", [])

    # Synthesize the verdict using the available Data
    chain = synthesis_verdict_prompt | llm | StrOutputParser()

    try:
        verdict = chain.invoke(
            {
                "user_query": user_query,
                "doct_analysis": doctrinal,
                "risks": risk,
                "precedents": precedents,
                "remediations": remediation,
                "previous_chats": messages,
                "analysis_units": analysis_units,
            }
        )
        return {"draft_verdict": verdict, "current_step": "synthesize_verdict"}
    except Exception as e:
        return {
            "draft_verdict": f"Error synthesizing: {e}",
            "current_step": "synthesize_verdict",
        }


#  9. Consistency Auditor
def consistency_auditor_and_cite(state: AgentState) -> dict:
    """
    A node that checks for consistency in the generated verdict and provides citations from the retrieved documents.

    Parameters:
    state (AgentState): The current state of the agent.

    Returns:
    dict: A dictionary containing the citations, consistency score, needs review flag, and the current step.
    """
    NODE_NAME = "consistency_auditor_and_cite"
    print(f"Node: {NODE_NAME}")

    # Block Execution in avoided by the decompose_to_analysis units node
    if is_node_blocked(state, NODE_NAME):
        print("NODE consistency_auditor_and_cite → blocked")
        return {
            "citations": [],
            "needs_review": False,
            "current_step": "consistency_auditor_and_cite",
        }

    draft = state.get("draft_verdict", "")
    retrieved_docs = state.get("retrieved_docs", [])
    risks = state.get("risk_assessment", {})

    # Get the citations from the retrieved documents
    citations = get_citations(retrieved_docs)

    # Audit the consistency of the generated verdict
    chain = consistency_auditor_and_cite_prompt | llm | JsonOutputParser()
    try:
        audit = chain.invoke({"draft": draft, "count": citations, "risks": risks})
        needs_review = (
            audit.get("contradiction_score", 0) > 50
            or audit.get("confidence", 100) < 50
        )
        return {
            "citations": citations,
            "consistency_score": audit.get("confidence", 0),
            "needs_review": needs_review,
            "review_count": state.get("review_count", 0) + (1 if needs_review else 0),
            "current_step": "consistency_auditor_and_cite",
        }
    except Exception as e:
        print(f"Audit error: {e}")
        return {"needs_review": False, "current_step": "consistency_auditor_and_cite"}


#  10. Finalize Response
def finalize_and_summarise_response(state: AgentState) -> dict:
    """
    Finalizes the response by adding references and a summary of the verdict.
    If the chat history is too long, it summarizes the chat history.

    Parameters:
    state (AgentState): The current state of the agent.

    Returns:
    dict: A dictionary containing the final response, the current step, and the messages to remove and add.
    """

    NODE_NAME = "finalize_and_summarise_response"
    print(f"Node: {NODE_NAME}")

    verdict = state.get("draft_verdict", "")
    citations = state.get("citations", [])
    messages = state.get("messages", [])

    # Format the references and add them to the response
    ref_text = (
        "\n\n## References\n"
        + "\n".join([f"- {c['label']}: {c['source']}" for c in citations])
        if citations
        else ""
    )

    response = f"{verdict}\n{ref_text}\n\n\n*AI-generated legal analysis.*"

    # Summarize the verdict and chat history to reduce the Token consuption in future calls
    summarise_verdict_chain = summarise_verdict_prompt | llm | StrOutputParser()
    verdict_summary = summarise_verdict_chain.invoke({"verdict": verdict})
    removemessages = []

    if len(messages) > 6:
        summarise_chat_chain = summarise_chat_prompt | llm | StrOutputParser()
        summary_chat = summarise_chat_chain.invoke({"chat_history": messages})
        removemessages = [
            RemoveMessage(id=str(msg.id)) for msg in state["messages"]
        ] + [AIMessage(summary_chat)]

    print("COMPLETED")
    return {
        "final_response": response,
        "current_step": "finalize_response",
        "messages": removemessages
        + [HumanMessage(state["user_query"]), AIMessage(verdict_summary.strip())],
    }
