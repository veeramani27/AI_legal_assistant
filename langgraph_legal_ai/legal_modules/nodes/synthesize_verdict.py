# LangChain / LangGraph Core
from langchain_core.output_parsers import StrOutputParser
from legal_modules.node_helpers import *
from legal_modules.prompts import *
from legal_modules.setup import llm
from legal_modules.state import AgentState


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
