import os
from dotenv import load_dotenv
import traceback
import re
import unicodedata
import multiprocessing

from langchain_google_community import GCSFileLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from google.cloud import storage

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID")
GCS_PREFIX = os.getenv("GCS_PREFIX")
VECTOR_STORE_FOLDER = "med_article_vdb0406"

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 100

load_dotenv()
GCP_CREDENTIAL_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if GCP_CREDENTIAL_JSON:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIAL_JSON
    print(f"[INFO] Using GCS credentials from: {GCP_CREDENTIAL_JSON}")
else:
    print("[WARNING] GOOGLE_APPLICATION_CREDENTIALS not found in .env. SDK might use default ADC.")

if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "test_key_not_functional":
    print("[ERROR] OPENAI_API_KEY must be set with a valid key to use OpenAI embeddings.")
    exit()
else:
    print("[INFO] OPENAI_API_KEY is set. OpenAI Embeddings will be used.")

def remove_repeated_substrings_custom(text, pattern_dot=None, pattern_comma=None, pattern_semicolon=None):
    if pattern_dot is None: pattern_dot = r'\.{2,}'
    if pattern_comma is None: pattern_comma = r',{2,}'
    if pattern_semicolon is None: pattern_semicolon = r';{2,}'
    text = re.sub(pattern_dot, '.', text)
    text = re.sub(pattern_comma, ',', text)
    text = re.sub(pattern_semicolon, ';', text)
    text = re.sub(r'\s*\.\s*', '. ', text)
    text = re.sub(r'\s*,\s*', ', ', text)
    text = re.sub(r'\s*;\s*', '; ', text)
    text = re.sub(r'\s*:\s*', ': ', text)
    text = re.sub(r'\s*\?\s*', '? ', text)
    text = re.sub(r'\s*!\s*', '! ', text)
    return text.strip()

def remove_extra_spaces_custom(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.replace(' \n', '\n')
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = '\n'.join([line.strip() for line in text.splitlines() if line.strip()])
    return text.strip()

def preprocess_text_custom(text: str) -> str:
    if not isinstance(text, str):
        # print(f"[WARNING] preprocess_text_custom received non-string input: {type(text)}. Returning empty string.") # Can be verbose
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = remove_extra_spaces_custom(text)
    text = remove_repeated_substrings_custom(text)
    text = re.sub(r"XSL-FO\s+RenderX", "", text, flags=re.IGNORECASE)
    text = re.sub(r"JMIR Res Protoc \d{4} vol\. \d+ e\d+ p\. \d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(page number not for citation purposes\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"J Med Internet Res \d{4} vol\. \d+ iss\. \d+ e\d+ p\. \d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r'[ \t]+', ' ', text).strip()
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def apply_custom_text_cleaning_to_documents(document_chunks: list, pdf_name_for_debug: str = "Unknown PDF") -> list:
    cleaned_docs = []
    for i, chunk_document in enumerate(document_chunks):
        try:
            new_chunk_document = chunk_document.model_copy()
        except AttributeError:
            new_chunk_document = chunk_document.copy()
        original_content = new_chunk_document.page_content
        cleaned_content = preprocess_text_custom(original_content)
        new_chunk_document.page_content = cleaned_content
        cleaned_docs.append(new_chunk_document)
    return cleaned_docs

def pypdf_loader_func(temporary_local_path: str) -> PyPDFLoader:
    loader_instance = PyPDFLoader(temporary_local_path)
    return loader_instance

def process_single_pdf_worker(args_tuple):
    """
    Loads, chunks, and cleans a single PDF file.
    Designed to be called by multiprocessing.Pool.
    """
    blob_name, gcs_project_id, gcs_bucket_name, chunk_size, chunk_overlap = args_tuple
    print(f"[WORKER-{os.getpid()}] Processing: {blob_name}")

    file_loader = GCSFileLoader(
        project_name=gcs_project_id,
        bucket=gcs_bucket_name,
        blob=blob_name,
        loader_func=pypdf_loader_func
    )
    try:
        documents = file_loader.load()
        if not documents:
            print(f"[WORKER-{os.getpid()}] No documents loaded from {blob_name}.")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True
        )
        texts_chunks = text_splitter.split_documents(documents)
        if not texts_chunks:
            print(f"[WORKER-{os.getpid()}] No text chunks created for {blob_name}.")
            return []
        
        cleaned_chunks_for_pdf = apply_custom_text_cleaning_to_documents(texts_chunks, blob_name)
        non_empty_cleaned_chunks = [doc for doc in cleaned_chunks_for_pdf if doc.page_content.strip()]

        if non_empty_cleaned_chunks:
            return non_empty_cleaned_chunks
        else:
            print(f"[WORKER-{os.getpid()}] All chunks from {blob_name} became empty after cleaning.")
            return []
            
    except Exception as e:
        print(f"[WORKER-{os.getpid()}] ERROR processing file {blob_name}: {e}")
        return []


def encode_all_pdfs_from_gcs_directory(
    gcs_project_id: str,
    gcs_bucket_name: str,
    gcs_directory_prefix: str,
    output_vector_store_folder: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
):
    print(f"[INFO] Starting encoding for all PDFs from GCS directory: gs://{gcs_bucket_name}/{gcs_directory_prefix}")

    storage_client = storage.Client(project=gcs_project_id if gcs_project_id else None)
    blobs = list(storage_client.list_blobs(gcs_bucket_name, prefix=gcs_directory_prefix))
    pdf_blobs_names = [
        blob.name for blob in blobs 
        if blob.name.lower().endswith(".pdf") and \
           (not gcs_directory_prefix.endswith('/') or blob.name != gcs_directory_prefix)
    ]

    if not pdf_blobs_names:
        print(f"[WARNING] No PDF files found in 'gs://{gcs_bucket_name}/{gcs_directory_prefix}'. Halting process.")
        return

    print(f"[INFO] Found {len(pdf_blobs_names)} PDF files to process in parallel.")
    
    tasks = [(name, gcs_project_id, 
              gcs_bucket_name, 
              chunk_size, 
              chunk_overlap) for name in pdf_blobs_names]
    
    all_processed_chunks = []
    # Using multiprocessing.Pool to process PDFs in parallel
    # The number of processes will default to os.cpu_count().
    # For an M1 Air (8 cores), this might be 4-8 processes.
    # You can specify processes=N in Pool(processes=N) if needed.
    # e.g., processes=max(1, os.cpu_count() // 2) to be less aggressive
    num_processes = min(len(pdf_blobs_names), os.cpu_count() or 1) # Don't use more processes than files or CPUs
    print(f"[INFO] Using {num_processes} parallel processes for PDF preprocessing.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_single_pdf_worker, tasks)
    
    for pdf_chunk_list in results:
        if pdf_chunk_list:
            all_processed_chunks.extend(pdf_chunk_list)

    if not all_processed_chunks:
        print("[ERROR] No processable text chunks found after processing all PDFs. Halting.")
        return

    print(f"\n[INFO] Total non-empty cleaned chunks from all PDFs to embed: {len(all_processed_chunks)}")

    # Create Embedding Model (OpenAI as requested)
    embeddings_model = None
    try:
        print("[INFO] Creating embedding model using OpenAI...")
        embeddings_model = OpenAIEmbeddings()
    except Exception as e:
        print(f"[ERROR] Failed to initialize OpenAI embedding model: {e}.")
        traceback.print_exc()
        return

    # Create FAISS Vector Store
    vectorstore = None
    if embeddings_model:
        print("[INFO] Creating FAISS vector store...")
        try:
            # FAISS.from_documents will call embeddings_model.embed_documents()
            # which typically batches requests to OpenAI efficiently.
            vectorstore = FAISS.from_documents(all_processed_chunks, embeddings_model)
            print("[INFO] FAISS vector store created successfully.")
            print(f"[INFO] Number of vectors in FAISS: {vectorstore.index.ntotal}")
        except Exception as e:
            print(f"[ERROR] Failed to create FAISS vector store: {e}")
            traceback.print_exc()
            return
    else:
        print("[WARNING] Embedding model not available. Skipping FAISS vector store creation.")
        return

    if vectorstore:
        print(f"\n[INFO] Saving vector store to: ./{output_vector_store_folder}")
        try:
            vectorstore.save_local(output_vector_store_folder)
            print(f"[INFO] Vector store saved successfully to folder: {output_vector_store_folder}")
        except Exception as e:
            print(f"[ERROR] Failed to save vector store: {e}")
            traceback.print_exc()
    else:
        print("[WARNING] No vector store to save.")

if __name__ == "__main__":
    
    multiprocessing.freeze_support() 

    if not GCP_CREDENTIAL_JSON:
        print("[ERROR] GOOGLE_APPLICATION_CREDENTIALS must be set to access GCS. Halting process.")
    elif not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "test_key_not_functional":
        print("[ERROR] OpenAI API key is missing or invalid. Halting process.")
    else:
        print(f"\n[INFO] Starting PDF processing and vector store creation...")
        print(f"[INFO] Target GCS Directory: gs://{GCS_BUCKET_NAME}/{GCS_PREFIX}")
        print(f"[INFO] Using OpenAI Embeddings.")

        encode_all_pdfs_from_gcs_directory(
            gcs_project_id=GCS_PROJECT_ID,
            gcs_bucket_name=GCS_BUCKET_NAME,
            gcs_directory_prefix=GCS_PREFIX,
            output_vector_store_folder=VECTOR_STORE_FOLDER,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        print("\n[INFO] Process completed.")