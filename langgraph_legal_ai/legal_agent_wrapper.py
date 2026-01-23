import uuid
from typing import Optional

import uvicorn
from core_graph import app, chain, langfuse_handler
from fastapi import Body, FastAPI
from pydantic import BaseModel

api = FastAPI()


class GraphRequest(BaseModel):
    query: str
    doc_path: Optional[str] = None
    thread_id: Optional[str] = None


class SummariseRequest(BaseModel):
    query: str
    response: str


@api.post("/run-legal-graph")
def run_legal_graph(payload: GraphRequest):
    """
    Run the legal graph with the given query and optional document path.

    Parameters:
    payload (GraphRequest): A GraphRequest object containing the query and optional document path.

    Returns:
    dict: A dictionary containing the status, thread_id, and result of running the legal graph.
    """
    thread_id = payload.thread_id or str(uuid.uuid4())

    graph_input = {
        "input_query": payload.query,
    }

    if payload.doc_path:
        print(payload.doc_path)
        graph_input["document_path"] = payload.doc_path

    result = app.invoke(
        graph_input,
        config={
            "configurable": {"thread_id": thread_id},
            "callbacks": [langfuse_handler],
        },
    )

    return {
        "status": "success",
        "thread_id": thread_id,
        "result": result,
    }


@api.post("/summarise")
def summarise(request: SummariseRequest):
    """
    Summarise the legal analysis of a user query.

    Parameters:
    request (SummariseRequest): A SummariseRequest object containing the user query and legal analysis.

    Returns:
    dict: A dictionary containing the status and result of summarising the legal analysis.
    """
    chain_input = {"user_query": request.query, "legal_analysis": request.response}

    result = chain.invoke(chain_input)

    return {
        "status": "success",
        "result": result,
    }
