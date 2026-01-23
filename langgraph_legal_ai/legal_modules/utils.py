#
# 3. HELPER FUNCTIONS
#

from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from legal_modules.setup import embeddings


def get_user_doc_collection_name(doc_path: str) -> str:
    """Generate a consistent hash-based collection name for user docs."""
    import hashlib

    hash_id = hashlib.md5(doc_path.encode()).hexdigest()[:8]
    return f"user_doc_{hash_id}"


def retrieve_filtered_documents(
    vectorstore: Chroma, query: str, k: int = 5, threshold: float = 0
) -> List[Document]:
    """Retrieve docs filtering by relevance score."""
    results = vectorstore.similarity_search_with_relevance_scores(query=query, k=k)
    return [doc for doc, score in results if score >= threshold]


def md(string):
    """Display Markdown."""
    from IPython.display import Markdown, display

    display(Markdown(string))


def visual(graph_compiled):
    """Display Graph."""
    from IPython.display import Image, display

    display(Image(graph_compiled.get_graph().draw_mermaid_png()))


def delete_doc_from_collection(persist_directory: str, user_doc_collection_name: str):
    """
    Deletes all documents from a ChromaDB collection.
    """
    try:
        db_local = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=user_doc_collection_name,
        )
        ids = db_local._collection.get(include=[])["ids"]
        db_local.delete(ids=ids)
        print(
            f"Chroma DB {db_local._collection_name} has {db_local._collection.count()} documents."
        )
    except:
        print("Chroma DB already empty")
