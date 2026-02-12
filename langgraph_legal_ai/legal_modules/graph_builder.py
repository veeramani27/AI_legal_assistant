#
# 7. GRAPH BUILDING
#

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from legal_modules.nodes.compliance_and_loophole_validator import \
    compliance_and_loophole_validator
from legal_modules.nodes.consistency_auditor_and_cite import \
    consistency_auditor_and_cite
from legal_modules.nodes.decompose_to_analysis_units import \
    decompose_to_analysis_units
from legal_modules.nodes.finalize_and_summarise_response import \
    finalize_and_summarise_response
from legal_modules.nodes.ingest_document_if_needed import \
    ingest_document_if_needed
from legal_modules.nodes.parallel_join_gate import parallel_join_gate
from legal_modules.nodes.precedent_matcher import precedent_matcher
from legal_modules.nodes.retriever import retriever
from legal_modules.nodes.risk_and_remediation_assessor import \
    risk_and_remediation_assessor
from legal_modules.nodes.synthesize_verdict import synthesize_verdict
from legal_modules.state import AgentState


def build_legal_graph() -> StateGraph:
    """
    Builds the legal language graph. This graph takes in a natural language query and returns a verdict that is grounded in relevant legal provisions.
    The graph consists of several nodes, each of which performs a specific task. The nodes are connected by edges, which represent the flow of information between the nodes.
    """
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("ingest_document_if_needed", ingest_document_if_needed)
    workflow.add_node("decompose_to_analysis_units", decompose_to_analysis_units)
    workflow.add_node("retriever", retriever)
    workflow.add_node(
        "compliance_and_loophole_validator", compliance_and_loophole_validator
    )
    workflow.add_node("precedent_matcher", precedent_matcher)
    workflow.add_node("risk_and_remediation_assessor", risk_and_remediation_assessor)
    workflow.add_node("parallel_join_gate", parallel_join_gate)
    workflow.add_node("synthesize_verdict", synthesize_verdict)
    workflow.add_node("consistency_auditor_and_cite", consistency_auditor_and_cite)
    workflow.add_node(
        "finalize_and_summarise_response", finalize_and_summarise_response
    )

    # Edges
    workflow.add_edge(START, "ingest_document_if_needed")
    workflow.add_edge("ingest_document_if_needed", "decompose_to_analysis_units")

    # Conditional: Check if legal context
    workflow.add_conditional_edges(
        "decompose_to_analysis_units",
        lambda state: (
            "end"
            if not state.get("intent_classification", {}).get(
                "query_related_to_legal_context", True
            )
            else "continue"
        ),
        {"end": END, "continue": "retriever"},
    )

    # Parallel Fan-out
    workflow.add_edge("retriever", "compliance_and_loophole_validator")
    workflow.add_edge("retriever", "precedent_matcher")

    workflow.add_edge(
        "compliance_and_loophole_validator", "risk_and_remediation_assessor"
    )

    # Parallel Fan-in
    workflow.add_edge("risk_and_remediation_assessor", "parallel_join_gate")
    workflow.add_edge("precedent_matcher", "parallel_join_gate")

    workflow.add_conditional_edges(
        "parallel_join_gate",
        lambda state: "continue" if state.get("parallel_join_complete") else "wait",
        {"continue": "synthesize_verdict", "wait": "parallel_join_gate"},
    )

    workflow.add_edge("synthesize_verdict", "consistency_auditor_and_cite")

    # Review Loop
    def should_review(state: AgentState) -> str:
        if not state.get("needs_review"):
            return "proceed"
        if state.get("review_count", 0) >= state.get("max_review_count", 2):
            return "force_proceed"
        return "retry"

    workflow.add_conditional_edges(
        "consistency_auditor_and_cite",
        should_review,
        {
            "retry": "retriever",
            "force_proceed": "finalize_and_summarise_response",
            "proceed": "finalize_and_summarise_response",
        },
    )

    workflow.add_edge("finalize_and_summarise_response", END)

    return workflow


conn = sqlite3.connect("./db/checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# Compile App
app = build_legal_graph().compile(checkpointer=checkpointer)
