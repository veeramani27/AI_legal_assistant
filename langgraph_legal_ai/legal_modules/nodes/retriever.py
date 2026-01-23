# LangChain / LangGraph Core
from legal_modules.node_helpers import *
from legal_modules.setup import db
from legal_modules.state import AgentState


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
