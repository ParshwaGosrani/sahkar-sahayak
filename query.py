import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

genai.configure(api_key=api_key)
DB_FAISS_PATH = "vectorstore/db_faiss"

def ask_question(user_query):
    print("\n1. Loading local vector database...")
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    print("2. Searching for relevant official information...")
    docs = db.similarity_search(user_query, k=10)
    
    context_text = ""
    sources = []
    for doc in docs:
        context_text += f"\n{doc.page_content}\n"
        sources.append(doc.metadata.get('page', 'Unknown'))

    print("3. Connecting to Google AI...")
    
    print("3. Connecting to Google AI...")
    target_model = "gemini-3.6-flash"
    print(f"   -> Connected to: {target_model}")
    model = genai.GenerativeModel(target_model)

    print(f"\n--- Retrieved {len(docs)} Chunks Preview ---")
    for i, doc in enumerate(docs[:3]):
        print(f"[Chunk {i+1} - Page {doc.metadata.get('page', '?')}]: {doc.page_content[:150]}...\n")
    
    prompt = f"""
    You are 'SAHKAR SAHAYAK', an expert assistant for Indian farmers and cooperative members.
    Answer the user's question accurately using the provided official context below.
    Synthesize key points (such as financial support, risk coverage, modern agricultural practices, or stabilization of farmer income) if mentioned across the passages.
    If the context genuinely contains zero relevant information, reply: "I could not verify this information from the available official sources."
    Keep the answer clear, structured with bullet points, and reply in the EXACT SAME LANGUAGE as the user's question.

    CONTEXT:
    {context_text}

    USER QUESTION:
    {user_query}
    """

    response = model.generate_content(prompt)
    
    print("\n================ ANSWER ================\n")
    print(response.text)
    print("\n================ SOURCES ===============")
    
    unique_pages = list(set([str(int(p) + 1) if p != 'Unknown' else p for p in sources]))
    print(f"Extracted from official document, Pages: {', '.join(unique_pages)}")
    print("========================================\n")

if __name__ == "__main__":
    query = input("\nAsk a question about the document (English, Hindi, or Marathi): ")
    ask_question(query)