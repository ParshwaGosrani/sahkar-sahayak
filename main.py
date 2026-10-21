import os
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found in environment variables")

# Initialize Groq Client
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
print("Loading vector database and embedding model via Hugging Face API...")
embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    huggingfacehub_api_token=hf_token
)
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
print("Vector database ready.")

class QueryRequest(BaseModel):
    query: str
    language: str = "en"

class QueryResponse(BaseModel):
    answer: str
    response: str  # Included both keys to guarantee frontend compatibility
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
            
            # Extract both the file name and the page number
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

        # Prioritized list of actual supported active Groq models
        fallback_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ]
        
        response_text = None
        
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
                    max_tokens=2048,
                )
                response_text = response.choices[0].message.content
                break # Exit the loop immediately if successful
            except Exception as e:
                print(f"Warning: {model_name} failed ({str(e)}). Switching models...")
                continue
                
        # If all models in the list fail
        if not response_text:
            error_msg = "The Sahkar Sahayak AI is currently experiencing unprecedented traffic across all backup servers. Please wait 30 seconds and try again."
            return QueryResponse(
                answer=error_msg,
                response=error_msg,
                sources=[]
            )

        unique_sources = sorted(list(set(sources)))
        
        return QueryResponse(
            answer=response_text,
            response=response_text,
            sources=unique_sources
        )
    
    except Exception as e:
        import traceback
        print("\n--- ERROR DETAILS ---")
        traceback.print_exc()
        print("---------------------\n")
        raise HTTPException(status_code=500, detail=str(e))