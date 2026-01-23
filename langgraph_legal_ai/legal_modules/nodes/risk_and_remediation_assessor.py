# LangChain / LangGraph Core
from langchain_core.output_parsers import JsonOutputParser
from legal_modules.node_helpers import *
from legal_modules.prompts import *
from legal_modules.setup import llm
from legal_modules.state import AgentState


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
