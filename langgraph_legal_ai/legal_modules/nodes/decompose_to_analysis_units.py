# LangChain / LangGraph Core
from langchain_core.output_parsers import JsonOutputParser
from legal_modules.node_helpers import *
from legal_modules.prompts import *
from legal_modules.setup import llm
from legal_modules.state import AgentState


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
