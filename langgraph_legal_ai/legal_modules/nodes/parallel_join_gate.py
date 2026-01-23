from legal_modules.state import AgentState


#  7. Parallel Join Gate
def parallel_join_gate(state: AgentState) -> dict:
    """
    A node that checks if all the parallel nodes (doctrinal, precedent, and remediation) are complete.
    If all the nodes are complete, it sets `parallel_join_complete` to True.

    Parameters:
    state (AgentState): The current state of the agent.

    Returns:
    dict: A dictionary containing `parallel_join_complete` and the status of different nodes.
    """

    NODE_NAME = "parallel_join_gate"
    print(f"Node: {NODE_NAME}")

    # If all the parallel nodes are complete then set `parallel_join_complete` to True
    if all(
        [
            state.get("doctrinal_done"),
            state.get("precedent_done"),
            state.get("remediation_done"),
        ]
    ):
        if not state.get("parallel_join_complete"):
            return {"parallel_join_complete": True}
    return {}
