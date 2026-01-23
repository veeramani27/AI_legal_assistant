from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
# LangChain / LangGraph Core
from langchain_core.output_parsers import StrOutputParser
from legal_modules.node_helpers import *
from legal_modules.prompts import *
from legal_modules.setup import llm
from legal_modules.state import AgentState


#  10. Finalize Response
def finalize_and_summarise_response(state: AgentState) -> dict:
    """
    Finalizes the response by adding references and a summary of the verdict.
    If the chat history is too long, it summarizes the chat history.

    Parameters:
    state (AgentState): The current state of the agent.

    Returns:
    dict: A dictionary containing the final response, the current step, and the messages to remove and add.
    """

    NODE_NAME = "finalize_and_summarise_response"
    print(f"Node: {NODE_NAME}")

    verdict = state.get("draft_verdict", "")
    citations = state.get("citations", [])
    messages = state.get("messages", [])

    # Format the references and add them to the response
    ref_text = (
        "\n\n## References\n"
        + "\n".join([f"- {c['label']}: {c['source']}" for c in citations])
        if citations
        else ""
    )

    response = f"{verdict}\n{ref_text}\n\n\n*AI-generated legal analysis.*"

    # Summarize the verdict and chat history to reduce the Token consuption in future calls
    summarise_verdict_chain = summarise_verdict_prompt | llm | StrOutputParser()
    verdict_summary = summarise_verdict_chain.invoke({"verdict": verdict})
    removemessages = []

    if len(messages) > 6:
        summarise_chat_chain = summarise_chat_prompt | llm | StrOutputParser()
        summary_chat = summarise_chat_chain.invoke({"chat_history": messages})
        removemessages = [
            RemoveMessage(id=str(msg.id)) for msg in state["messages"]
        ] + [AIMessage(summary_chat)]

    print("COMPLETED")
    return {
        "final_response": response,
        "current_step": "finalize_response",
        "messages": removemessages
        + [HumanMessage(state["user_query"]), AIMessage(verdict_summary.strip())],
    }
