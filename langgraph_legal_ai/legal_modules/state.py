#
# 4. STATE DEFINITION
#


from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from langgraph.graph.message import BaseMessage, add_messages


class AgentState(TypedDict):
    """
    The state of the agent.
    """

    # Inputs
    input_query: str
    document_path: Optional[str]
    document_text: Optional[str]

    # Processing
    user_query: str
    actions_needed: List[str]
    analysis_units: List[str]
    retrieved_docs: List[Dict[str, Any]]
    user_doc_collection: Optional[str]
    intent_classification: Optional[Dict[str, Any]]

    # Agent Outputs
    doctrinal_analysis: Optional[Dict[str, Any]]
    precedent_matches: Optional[List[Dict[str, Any]]]
    loophole_analysis: Optional[Dict[str, Any]]
    risk_assessment: Optional[Dict[str, Any]]
    remediation_suggestions: Optional[List[str]]
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Synthesis & Output
    draft_verdict: Optional[str]
    citations: Optional[List[Dict[str, Any]]]
    consistency_score: Optional[float]
    final_response: Optional[str]

    # Control Flow
    needs_review: bool
    doctrinal_done: Optional[bool]
    precedent_done: Optional[bool]
    remediation_done: Optional[bool]
    parallel_join_complete: Optional[bool]

    review_count: int
    max_review_count: int
    current_step: str
    error: Optional[str]
