import os
from langchain_chroma import Chroma
from app.config import VECTOR_DIR
from app.embeddings import get_embeddings

def get_vectorstore(documents=None, collection_name="default"):
    """
    Get or create a vectorstore
    
    Args:
        documents: Optional documents to vectorize
        collection_name: Name of the collection to use
    """
    os.makedirs(VECTOR_DIR, exist_ok=True)

    embeddings = get_embeddings()

    if documents:
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=VECTOR_DIR,
            collection_name=collection_name
        )

    return Chroma(
        embedding_function=embeddings,
        persist_directory=VECTOR_DIR,
        collection_name=collection_name
    )