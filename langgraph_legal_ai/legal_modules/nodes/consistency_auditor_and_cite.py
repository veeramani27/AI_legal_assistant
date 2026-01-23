# LangChain / LangGraph Core
from langchain_core.output_parsers import JsonOutputParser
from legal_modules.node_helpers import *
from legal_modules.prompts import *
from legal_modules.setup import llm
from legal_modules.state import AgentState


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
