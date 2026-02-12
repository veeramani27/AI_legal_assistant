import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from legal_modules.prompts import *
from legal_modules.setup import embeddings, llm
from legal_modules.tools import web_search_tool, websearch_llm
from legal_modules.utils import *


def is_node_blocked(state: dict, NODE_NAME: str) -> bool:
    """
    Checks if a node is blocked by checking if the node name is in the "actions_needed" list in the state.

    Args:
        state (dict): The current state of the agent.
        NODE_NAME (str): The name of the node to check.

    Returns:
        bool: True if the node is blocked, False otherwise.
    """
    return NODE_NAME not in state.get("actions_needed", [])


def chunk_and_save_to_chromadb(document_text: str, document_path: str):
    """
    Chunk a document into smaller pieces and save them to a ChromaDB collection.

    Parameters:
    document_text (str): The text of the document to chunk.
    document_path (str): The path to the document.

    Returns:
    str: The name of the ChromaDB collection where the document was saved.
    """

    # 1. Initialize Semantic Chunker
    text_splitter = SemanticChunker(
        embeddings=embeddings,  
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
    )

    semantic_docs = text_splitter.create_documents([document_text])

    # 2. Metadata Enrichment - Add Section Headers
    enriched_documents = []
    header_pattern = re.compile(
        r"^(Article|Section|Clause|\d+\.)\s.*", re.IGNORECASE | re.MULTILINE
    )

    for i, doc in enumerate(semantic_docs):
        content = doc.page_content

        header_match = header_pattern.match(content.strip())
        section_header = header_match.group(0) if header_match else "Clause"

        enriched_documents.append(
            Document(
                page_content=content,
                metadata={
                    "section_header": section_header,
                    "chunk_index": i,
                    "type": "contract_clause",
                    "source": document_path,
                },
            )
        )

    # documents = [
    #     Document(page_content=chunk, metadata={"source": document_path})
    #     for chunk in chunks
    # ]

    collection_name = get_user_doc_collection_name(document_path)

    # Create separate vectorstore for user doc
    user_vectorstore = Chroma.from_documents(
        documents=enriched_documents,
        collection_name=collection_name,
        persist_directory="./user-docs",
        embedding=embeddings,
    )

    print(f"Document saved to collection: {user_vectorstore._collection_name}")

    return user_vectorstore._collection_name


def get_analysis_units(
    result: dict, intent: str, document_text: str, user_doc_collection: str
):
    """
    Generate analysis units based on the user query, intent, and document text.

    If the intent is "general", the analysis unit is the user query.
    If the intent is "document_general", the analysis units are the chunks of the document text.
    If the intent is "document_specific", the analysis units are the relevant chunks of the document text retrieved from the user document ChromaDB collection.

    Parameters:
    result (dict): The result of the intent classification model.
    intent (str): The intent of the user query.
    document_text (str): The text of the document.
    user_doc_collection (str): The name of the user document ChromaDB collection.

    Returns:
    list: A list of analysis units.
    """
    analysis_units = []

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    user_query = result.get("optimised_query")
    if intent == "general":
        analysis_units = [user_query]
    elif intent == "document_general":
        analysis_units = (
            splitter.split_text(document_text) if document_text else [user_query]
        )
    elif intent == "document_specific":
        # Retrieve relevant chunks
        collection_name = user_doc_collection
        if collection_name and document_text:
            try:
                user_db = Chroma(
                    collection_name=collection_name,
                    persist_directory="./user-docs",
                    embedding_function=embeddings,
                )
                docs = retrieve_filtered_documents(user_db, user_query, k=5)
                analysis_units = [doc.page_content for doc in docs]
            except Exception as e:
                analysis_units = splitter.split_text(document_text)
        else:
            analysis_units = [user_query]
    else:
        analysis_units = [user_query]

    return analysis_units


def get_relevant_docs(query, analysis_units, db):
    """
    Retrieves relevant documents from the database based on the user query and analysis units.

    Parameters:
    query (str): The user query.
    analysis_units (list): A list of analysis units.
    db (Chroma): The Chroma database object.

    Returns:
    list: A list of relevant documents.
    """
    queries = [query] + analysis_units
    unique_docs = []
    seen = set()

    for q in queries:
        docs = retrieve_filtered_documents(db, q, k=5, threshold=0.1)
        for doc in docs:
            key = (doc.page_content.strip(), doc.metadata.get("source"))
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

    if not unique_docs:
        unique_docs = [
            Document(
                page_content="No relevant provisions found.",
                metadata={"source": "system"},
            )
        ]

    return unique_docs


def extract_loopholes(findings):
    """
    Extracts loophole information from the findings.

    Parameters:
    findings (list): A list of findings containing loophole information.

    Returns:
    list: A list of dictionaries containing the loophole type, description, and the corresponding clause summary.
    """
    return [
        {
            "clause_summary": f["clause"],
            "type": f["associated_loophole"]["type"],
            "description": f["associated_loophole"]["description"],
        }
        for f in findings
        if f.get("associated_loophole", {}).get("type") != "none"
    ]


def execute_search_tool(raw_llm_response):
    """
    Executes a search tool based on the tool calls in the raw_llm_response.
    Supports web search tool.

    Parameters:
    raw_llm_response (dict): The raw response from the LLM containing tool calls.

    Returns:
    str: The formatted web context string containing the search results.

    Raises:
    Exception: If there is an error executing the web search tool.
    """
    search_tool_map = {"web_search_tool": web_search_tool}
    web_context = ""

    for tool_call in raw_llm_response.tool_calls:
        selected_tool = search_tool_map.get(tool_call["name"])
        if selected_tool:
            try:
                # Execute the search
                tool_output = selected_tool.invoke(tool_call["args"])

                # Format Web Results
                search_results = tool_output.get("results", [])
                formatted_web_results = []
                for res in search_results:
                    formatted_web_results.append(
                        f"Web Source: {res.get('title')}\nURL: {res.get('url')}\nContent: {res.get('content', 'N/A')[:1000]}..."
                    )

                web_context = "\n\n".join(formatted_web_results)

            except Exception as e:
                print(f"Error executing web search tool: {e}")
    return web_context


def match_precedent(user_query: str, messages: list, local_case_context: str):
    """
    A node that matches the user query with relevant precedents from the database of legal cases.
    If needed, external resources (web search) are used to find relevant precedents.

    Parameters:
    user_query (str): The user query to be matched with precedents.
    messages (list): The previous chat messages.
    local_case_context (str): The context of local legal cases.

    Returns:
    dict: A dictionary containing the relevant precedents, the current step, and the status of different nodes.

    Raises:
    Exception: If there is an error executing the web search tool or processing the LLM response.
    """
    try:
        raw_llm_response = (precedent_matcher_prompt | websearch_llm).invoke(
            {
                "user_query": user_query,
                "messages": messages,
                "local_case_context": local_case_context,
            }
        )

        web_context = ""
        if hasattr(raw_llm_response, "tool_calls") and raw_llm_response.tool_calls:
            print("Precedent Matcher: Initiating Web Search for additional cases...")
            web_context = execute_search_tool(raw_llm_response)
        else:
            print("Precedent Matcher: Using local knowledge only.")

        if web_context:
            print(
                "Precedent Matcher: Web Search found additional cases and passsed to llm..."
            )

            matches = (
                final_precedent_matcher_prompt | llm | JsonOutputParser()
            ).invoke(
                {
                    "user_query": user_query,
                    "messages": messages,
                    "local_case_context": local_case_context,
                    "web_context": web_context,
                }
            )
        else:

            # Parsing the text content directly if it's already an answer
            if isinstance(raw_llm_response.content, str):
                matches = (precedent_matcher_prompt | llm | JsonOutputParser()).invoke(
                    {
                        "user_query": user_query,
                        "messages": messages,
                        "local_case_context": local_case_context,
                    }
                )
            else:
                matches = []

        return {"precedent_matches": matches[:3], "precedent_done": True}

    except Exception as e:
        print(f"Error in precedent node: {e}")
        return {"precedent_matches": [], "precedent_done": True}


def get_citations(retrieved_docs: str):
    """
    Retrieves citations from the retrieved documents.

    Parameters:
    retrieved_docs (list): A list of dictionaries containing the retrieved documents.

    Returns:
    list: A list of dictionaries containing the citations, where each citation contains the source, label, and excerpt.
    """
    citations = []
    seen = set()
    for doc in retrieved_docs[:5]:
        cite_data = (
            doc.metadata.get("case_name")
            or doc.metadata.get("section")
            or doc.metadata.get("source")
        )
        if cite_data and cite_data not in seen:
            seen.add(cite_data)
            citations.append(
                {
                    "source": doc.metadata.get("source"),
                    "label": cite_data,
                    "excerpt": doc.page_content[:200] + "...",
                }
            )

    return citations
