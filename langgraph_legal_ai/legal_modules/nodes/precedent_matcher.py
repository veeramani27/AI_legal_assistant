# LangChain / LangGraph Core
from legal_modules.node_helpers import *
from legal_modules.prompts import *
from legal_modules.state import AgentState


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
