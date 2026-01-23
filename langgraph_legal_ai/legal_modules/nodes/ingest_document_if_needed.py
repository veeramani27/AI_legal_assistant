from langchain_community.document_loaders import PyMuPDFLoader
from legal_modules.node_helpers import *
from legal_modules.state import AgentState


#  1. Ingestion
def ingest_document_if_needed(state: AgentState) -> dict:
    """
    Ingest document if needed.

    Checks if a document path is provided and if so, loads the document and chunks it into a ChromaDB collection.
    If the document text already exists in the state, it skips the ingestion step.

    Returns a dictionary with the following keys:
    - user_doc_collection: the name of the ChromaDB collection containing the ingested document
    - current_step: the name of the current step in the workflow
    - review_count: the number of times the document has been reviewed
    """

    NODE_NAME = "ingest_document_if_needed"
    print(f"Node: {NODE_NAME}")

    document_path = state.get("document_path")
    if not document_path:
        return {
            "user_doc_collection": None,
            "current_step": "ingest_document_if_needed",
            "review_count": 0,
        }

    # Check to avoid re-processing if text exist
    if state.get("document_text"):
        return {"current_step": "ingest_document_if_needed", "review_count": 0}

    try:
        loader = PyMuPDFLoader(document_path)
        docs = loader.load()
        document_text = "".join(doc.page_content for doc in docs)
    except Exception as e:
        return {
            "error": f"PDF parsing failed: {str(e)}",
            "current_step": "ingest_document_if_needed",
        }

    # Chunking & Embedding
    collection_name = chunk_and_save_to_chromadb(document_text, document_path)

    return {
        "document_text": document_text,
        "user_doc_collection": collection_name,
        "current_step": "ingest_document_if_needed",
        "review_count": 0,
    }
