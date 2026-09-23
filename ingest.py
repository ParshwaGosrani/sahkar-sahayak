import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DATA_DIR = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_vector_db():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory '{DATA_DIR}' not found. Please create it and add your PDFs.")
        return

    print(f"Loading all PDF documents from '{DATA_DIR}'...")
    # This loader reads every PDF in the folder automatically
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()
    print(f"Loaded {len(documents)} total pages from all documents.")

    if len(documents) == 0:
        print("No PDFs found in the data directory!")
        return

    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    print("Generating local multilingual embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    print("Saving to FAISS vector store...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"Vector store successfully saved to '{DB_FAISS_PATH}'.")

if __name__ == "__main__":
    create_vector_db()