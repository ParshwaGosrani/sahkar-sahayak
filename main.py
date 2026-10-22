import os
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Initialize Groq Client for fast answer generation
client = Groq(api_key=api_key)

DB_FAISS_PATH = "vectorstore/db_faiss"

# Initialize FastAPI app
app = FastAPI(
    title="Sahkar Sahayak API",
    description="Multilingual AI Assistant for Farmers & Cooperative Schemes",
    version="1.0.0"
)

# Enable CORS for React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache loaded vector database and model on startup
print("Loading vector database and embedding model via free Gemini API...")

# Initializing Gemini Embeddings (the "Librarian")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
print("Vector database ready.")

class QueryRequest(BaseModel):
    query: str
    language: str = "en"

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@app.get("/")
def health_check():
    return {"status": "online", "service": "Sahkar Sahayak API"}

@app.post("/api/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Retrieve relevant chunks from FAISS
        docs = db.similarity_search(request.query, k=8)
        
        context_text = ""
        sources = []
        for doc in docs:
            context_text += f"\n{doc.page_content}\n"
            
            source_file = doc.metadata.get("source", "Unknown")
            file_name = os.path.basename(source_file)
            page = doc.metadata.get("page", 0)
            
            sources.append(f"{file_name} - Page {int(page) + 1}")

        # Exact Same Few-Shot Prompt Engineering
        prompt = f"""
        You are 'SAHKAR SAHAYAK', an expert assistant for Indian farmers and cooperative members.
        Answer the user's question accurately using the provided official context below.
        Synthesize key points (such as financial support, risk coverage, modern agricultural practices, or stabilization of farmer income) if mentioned across the passages.
        If the context genuinely contains zero relevant information, reply: "I could not verify this information from the available official sources."
        Keep the answer clear, structured with bullet points, and reply in the EXACT SAME LANGUAGE as the user's question.

        --- FEW-SHOT EXAMPLES ---
        User: What are the premium rates for Kharif crops under PMFBY?
        AI: Based on the official guidelines, the premium rates for Kharif crops under PMFBY are:
        * Maximum 2% of the sum insured for food and oilseed crops.
        * For commercial/horticultural crops, it can be up to 5%.
        
        User: Can I claim PMFBY if my tractor breaks down?
        AI: I could not verify this information from the available official sources. PMFBY strictly covers crop loss due to non-preventable natural risks.
        -------------------------

        CONTEXT:
        {context_text}

        USER QUESTION:
        {request.query}
        """

        # Prioritized list of active models accessible on this account tier
        fallback_models = [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b"
        ]
        
        response_text = None
        last_error = "No error recorded."
        
        # Try each model until one succeeds
        for model_name in fallback_models:
            try:
                print(f"Attempting generation with {model_name}...")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                )
                
                content = response.choices[0].message.content
                
                if content and content.strip():
                    response_text = content
                    break 
                else:
                    last_error = "Model returned an empty response."
                    print(f"Warning: {model_name} returned empty text. Switching...")
                    continue
                    
            except Exception as e:
                last_error = str(e)
                print(f"Warning: {model_name} failed ({last_error}). Switching models...")
                continue
                
        # If all models in the list fail or return empty
        if not response_text:
            return QueryResponse(
                answer=f"System Error: {last_error}",
                sources=[]
            )

        unique_sources = sorted(list(set(sources)))
        
        return QueryResponse(
            answer=response_text,
            sources=unique_sources
        )
    
    except Exception as e:
        import traceback
        print("\n--- ERROR DETAILS ---")
        traceback.print_exc()
        print("---------------------\n")
        raise HTTPException(status_code=500, detail=str(e))