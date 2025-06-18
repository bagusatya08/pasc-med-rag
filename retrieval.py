import os
import traceback
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangchainDocument

DEFAULT_VECTOR_STORE_FOLDER = "med_article_vdb0406"
DEFAULT_NUM_RESULTS = 3

load_dotenv()

embeddings_model_global: Optional[OpenAIEmbeddings] = None
vector_db_global: Optional[FAISS] = None
is_initialized = False

def initialize_retriever_resources(vector_store_path: Optional[str] = None) -> bool:
    """
    Initializes the OpenAI Embeddings model and loads the FAISS vector store.
    This function should be called once when the application using this module starts.

    Args:
        vector_store_path (Optional[str]): Path to the FAISS vector store folder.
                                           If None, uses VECTOR_STORE_PATH from .env or DEFAULT_VECTOR_STORE_FOLDER.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    global embeddings_model_global, vector_db_global, is_initialized

    if is_initialized:
        print("[RETRIEVER_MODULE INFO] Resources already initialized.")
        return True

    print("[RETRIEVER_MODULE INFO] Initializing retriever resources...")

    # 1. Check and Initialize OpenAI Embeddings Model
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "test_key_not_functional":
        print("[RETRIEVER_MODULE ERROR] OPENAI_API_KEY must be set with a valid key.")
        return False
    print("[RETRIEVER_MODULE INFO] OPENAI_API_KEY is set.")

    try:
        embeddings_model_global = OpenAIEmbeddings()
        print("[RETRIEVER_MODULE INFO] OpenAI Embeddings model initialized.")
    except Exception as e:
        print(f"[RETRIEVER_MODULE ERROR] Failed to initialize OpenAI embeddings: {e}")
        traceback.print_exc()
        return False

    # 2. Load FAISS Vector Store
    store_path_to_load = vector_store_path or os.getenv("VECTOR_STORE_PATH", DEFAULT_VECTOR_STORE_FOLDER)
    
    if not os.path.exists(store_path_to_load):
        print(f"[RETRIEVER_MODULE ERROR] Vector store folder not found: {store_path_to_load}")
        return False

    print(f"[RETRIEVER_MODULE INFO] Loading FAISS vector store from: {store_path_to_load}...")
    try:
        vector_db_global = FAISS.load_local(
            store_path_to_load,
            embeddings_model_global,
            allow_dangerous_deserialization=True
        )
        print("[RETRIEVER_MODULE INFO] FAISS vector store loaded successfully.")
        if hasattr(vector_db_global, 'index') and vector_db_global.index is not None:
            print(f"[RETRIEVER_MODULE INFO] Number of vectors in loaded store: {vector_db_global.index.ntotal}")
        else:
            print("[RETRIEVER_MODULE INFO] Loaded store, but index info not readily available or store might be empty.")
        
        is_initialized = True
        return True
    except Exception as e:
        print(f"[RETRIEVER_MODULE ERROR] Failed to load FAISS vector store from {store_path_to_load}: {e}")
        traceback.print_exc()
        return False

def get_relevant_chunks(query: str, k: int = DEFAULT_NUM_RESULTS) -> List[Tuple[LangchainDocument, float]]:
    """
    Retrieves relevant document chunks and their scores for a given query.

    Args:
        query (str): The search query string.
        k (int): The number of relevant documents to retrieve.

    Returns:
        List[Tuple[LangchainDocument, float]]: A list of (document, score) tuples.
                                              Returns an empty list if resources are not initialized or an error occurs.
    
    Raises:
        RuntimeError: If retriever resources (embeddings model or vector store) are not initialized.
    """
    global vector_db_global, embeddings_model_global, is_initialized

    if not is_initialized or not vector_db_global or not embeddings_model_global:
        print("[RETRIEVER_MODULE ERROR] Retriever resources are not initialized. Call initialize_retriever_resources() first.")
        # Or raise an exception:
        raise RuntimeError("Retriever resources are not initialized. Call initialize_retriever_resources() first.")


    print(f"\n[RETRIEVER_MODULE INFO] Retrieving top {k} relevant documents with scores for query: \"{query}\"")
    try:
        results_with_scores = vector_db_global.similarity_search_with_relevance_scores(query, k=k)
        print(f"[RETRIEVER_MODULE INFO] Retrieved {len(results_with_scores)} documents with scores.")
        return results_with_scores
    except Exception as e:
        print(f"[RETRIEVER_MODULE ERROR] An error occurred during similarity search: {e}")
        traceback.print_exc()
        return [] # Return empty list on error

if __name__ == "__main__":
    print("--- Testing retriever_module.py ---")
    if initialize_retriever_resources(): # Uses default/env path for store
        print("\n--- Initialization Successful ---")
        
        test_query = "I have a headache, what should i consume to handle this?"
        test_k = 2
        
        retrieved_data = get_relevant_chunks(test_query, k=test_k)
        
        if retrieved_data:
            print(f"\n--- Test Retrieval Results for query: '{test_query}' (k={test_k}) ---")
            for i, (doc, score) in enumerate(retrieved_data):
                print(f"\n--- Document {i+1} (Relevance Score: {score:.4f}) ---")
                source = doc.metadata.get('source', 'N/A') if hasattr(doc, 'metadata') else 'N/A'
                page_num = doc.metadata.get('page', -1) if hasattr(doc, 'metadata') else -1
                page = str(page_num + 1) if page_num != -1 else 'N/A'
                print(f"Source: {source}, Page: {page}")
                content_snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                print(f"Content Snippet:\n{content_snippet}")
        else:
            print(f"\nNo documents retrieved for test query: '{test_query}'")
    else:
        print("\n--- Initialization Failed ---")
