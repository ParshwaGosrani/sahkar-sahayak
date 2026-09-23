# Sahkar Sahayak 🌾

**Sahkar Sahayak** is a full-stack, multilingual AI assistant built to democratize access to complex Indian agricultural policies and cooperative frameworks. Designed primarily for farmers and cooperative members, it uses a Retrieval-Augmented Generation (RAG) architecture to instantly answer queries regarding schemes like the Pradhan Mantri Fasal Bima Yojana (PMFBY) and the Multi-State Co-operative Societies Act with exact page-level citations.

## ✨ Features
* **Multilingual Support:** Processes and responds to queries in English, Hindi, and Marathi.
* **Zero Hallucination:** Utilizes advanced Prompt Engineering (Few-Shot prompting, Temperature=0.2) to strictly bind the AI's knowledge to official government documents.
* **Precise Citations:** Extracts the exact source file and page number for every claim made by the AI.
* **Local Vector Retrieval:** Employs a local FAISS database for lightning-fast, offline semantic similarity searches.

## 🛠️ Tech Stack
* **Backend:** Python, FastAPI, Uvicorn
* **AI/ML:** LangChain, FAISS, HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`), Google Gemini API (`gemini-3.6-flash`)
* **Frontend:** React, Vite, Tailwind CSS, Lucide React, React Markdown

## 🚀 Getting Started

### Prerequisites
* Python 3.12+
* Node.js (v18+)
* A Google Gemini API Key

### 1. Clone the Repository
```bash
git clone [https://github.com/ParshwaGosrani/sahkar-sahayak.git](https://github.com/ParshwaGosrani/sahkar-sahayak.git)
cd sahkar-sahayak
```
### 2. Backend Setup
Create a virtual environment and install the required Python packages.
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```
Create a .env file in the root directory and add your Gemini API key:
GEMINI_API_KEY=your_actual_api_key_here

Start the FastAPI server:
```bash
uvicorn main:app --reload
```
###3. Frontend Setup
Open a new terminal window, navigate to the frontend directory, and install the dependencies.
```bash
cd frontend
npm install
```
Start the Vite development server:

```Bash
npm run dev
```
The frontend will be available at http://localhost:5173.

## 📁 Project Structure

```text
sahkar-sahayak/
├── main.py                  # FastAPI application and RAG pipeline
├── ingest.py                # Script to convert PDFs into FAISS vector embeddings
├── .env                     # Environment variables (API Keys)
├── vectorstore/             # Local FAISS database files (generated)
├── data/                    # Directory containing official government PDFs
└── frontend/                # React Vite application
    ├── src/
    │   └── App.jsx          # Main Chat UI and API integration
    ├── package.json
    └── tailwind.config.js
```

## ⚙️ How it Works
1. **Document Ingestion:** `ingest.py` reads official PDFs, splits them into 1000-character chunks, and generates high-dimensional embeddings using a HuggingFace transformer model.
2. **Retrieval:** When a user asks a question, FastAPI vectorizes the query and performs a Cosine Similarity search in the FAISS database to find the most relevant policy clauses.
3. **Generation:** The retrieved chunks are injected into a Few-Shot prompt template and sent to Gemini, which synthesizes a formatted, cited response.

## 👤 Author
**Parshwa Gosrani** 
* GitHub: [@ParshwaGosrani](https://github.com/ParshwaGosrani)
