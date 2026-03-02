import os
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# BAAI/bge-small-en-v1.5 is ~24MB via ONNX — no PyTorch needed
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DB_DIR = "faiss_indices"

# Initialize embeddings globally to avoid reloading
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings

def get_faiss_index_path(subject: str) -> str:
    """Return the directory path for a specific subject's FAISS index."""
    return os.path.join(DB_DIR, subject.replace(" ", "_").lower())

def add_to_vector_store(subject: str, chunks: list[str], source: str = "Unknown"):
    """Converts text chunks to embeddings and adds them to the FAISS DB for the subject."""
    if not chunks:
        return False
        
    embeddings = get_embeddings()
    docs = [Document(page_content=chunk, metadata={"source": source}) for chunk in chunks]
    index_path = get_faiss_index_path(subject)
    
    if os.path.exists(index_path):
        # Load existing and add new docs
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(docs)
    else:
        # Create new index
        db = FAISS.from_documents(docs, embeddings)
        os.makedirs(DB_DIR, exist_ok=True)
        
    # Save back to disk
    db.save_local(index_path)
    return True

def retrieve_top_k(subject: str, query: str, k: int = 5):
    """Retrieves the top k most relevant chunks for a given query and subject."""
    index_path = get_faiss_index_path(subject)
    
    if not os.path.exists(index_path):
        return [] # No knowledge base for this subject yet
        
    embeddings = get_embeddings()
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    # Return documents with similarity scores
    results = db.similarity_search_with_score(query, k=k)
    return results
