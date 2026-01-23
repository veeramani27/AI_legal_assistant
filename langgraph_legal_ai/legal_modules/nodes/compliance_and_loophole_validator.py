# LangChain / LangGraph Core
from langchain_core.output_parsers import JsonOutputParser
from legal_modules.node_helpers import *
from legal_modules.prompts import *
from legal_modules.setup import llm
from legal_modules.state import AgentState


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
