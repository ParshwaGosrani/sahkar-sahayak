import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("GEMINI_API_KEY not found in .env")

print("Loading PDFs...")
# CHANGE "data" IF YOUR PDFS ARE IN A DIFFERENT FOLDER
loader = PyPDFDirectoryLoader("data") 
documents = loader.load()

print(f"Loaded {len(documents)} pages. Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

print("Generating free Gemini Embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

db = None
batch_size = 80

for i in range(0, len(chunks), batch_size):
    batch = chunks[i : i + batch_size]
    print(f"Processing batch {i} to {i + len(batch)} of {len(chunks)}...")
    
    if db is None:
        db = FAISS.from_documents(batch, embeddings)
    else:
        db.add_documents(batch)
        
    # SAVE PROGRESS AFTER EVERY BATCH
    db.save_local("vectorstore/db_faiss")
    print("-> Progress saved successfully.")
        
    if i + batch_size < len(chunks):
        print("Pausing for 60 seconds to respect free-tier limits...")
        time.sleep(60)

print("Database successfully rebuilt! You are ready to deploy.")