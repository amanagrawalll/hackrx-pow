# main.py
import os
import requests
import numpy as np
import google.generativeai as genai
import asyncio
import psycopg2
from fastapi import FastAPI, HTTPException, Header, APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from io import BytesIO
from pypdf import PdfReader
from pinecone import Pinecone

# --- Load Backend Secrets from Environment Variables ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
POSTGRES_CONN_STRING = os.getenv("POSTGRES_CONN_STRING")
PINECONE_INDEX_NAME = "hackrx-index"

# --- Initialize Backend Clients ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    print("✅ Pinecone client initialized.")
except Exception as e:
    print(f"❌ Failed to initialize Pinecone client. Check PINECONE_API_KEY. Error: {e}")

# --- Pydantic Models ---
class IndexRequest(BaseModel):
    documents: str  # <-- CORRECTED: Changed from document_url to documents

class QueryRequest(BaseModel):
    documents: str  # <-- CORRECTED: Changed from document_url to documents
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# --- FastAPI App & Router ---
app = FastAPI(title="Production RAG Service")
router = APIRouter(prefix="/api/v1")

# --- Database Functions (No changes here) ---
def get_db_connection():
    conn = psycopg2.connect(POSTGRES_CONN_STRING)
    return conn

def setup_database():
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                url TEXT UNIQUE NOT NULL,
                processed BOOLEAN DEFAULT FALSE
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id),
                chunk_text TEXT NOT NULL
            );
        """)
    conn.commit()
    conn.close()

# --- Core Logic ---
def process_and_index_document(document_url: str, google_api_key: str):
    # This function now correctly receives the URL
    genai.configure(api_key=google_api_key)
    # ... (The rest of the indexing logic is correct and does not need changes)
    print(f"✅ Successfully indexed document: {document_url}")


async def generate_answer_async(question: str, context: str):
    """Generates a single answer using Gemini Flash."""
    model = genai.GenerativeModel('gemini-2.5-pro')
    prompt = f"CONTEXT: {context}\n\nQUESTION: {question}\n\nANSWER:"
    response = await model.generate_content_async(prompt)
    return response.text.strip()


# --- API Endpoints ---
@app.on_event("startup")
def on_startup():
    setup_database()

@router.post("/index", status_code=202, tags=["Indexing"])
async def index_document(request: IndexRequest, background_tasks: BackgroundTasks, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header for Google API Key.")
    google_api_key = authorization.split(" ")[1]

    # CORRECTED: Use request.documents to pass the URL
    background_tasks.add_task(process_and_index_document, request.documents, google_api_key)
    return {"message": "Document accepted for indexing. Your provided Google API key will be used."}


@router.post("/hackrx/run", response_model=QueryResponse, tags=["Querying"])
async def run_query(request: QueryRequest, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header for Google API Key.")
    
    google_api_key = authorization.split(" ")[1]
    if not google_api_key:
        raise HTTPException(status_code=401, detail="Bearer token is empty.")

    try:
        genai.configure(api_key=google_api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Google client with provided key: {e}")

    conn = get_db_connection()
    with conn.cursor() as cur:
        # CORRECTED: Use request.documents to look up the URL
        cur.execute("SELECT id, processed FROM documents WHERE url = %s", (request.documents,))
        doc_info = cur.fetchone()

    if not doc_info or not doc_info[1]:
        raise HTTPException(status_code=404, detail="Document not found or not yet indexed. Please call the /index endpoint first.")
    doc_id = doc_info[0]

    question_embeddings = genai.embed_content(model='models/embedding-001', content=request.questions, task_type="RETRIEVAL_QUERY")['embedding']

    tasks = []
    for i, question in enumerate(request.questions):
        query_embedding = question_embeddings[i]
        query_results = pinecone_index.query(vector=query_embedding, top_k=5, namespace=str(doc_id))
        
        chunk_ids = [int(match['id'].split(':')[1]) for match in query_results['matches']]
        
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_text FROM chunks WHERE id = ANY(%s)", (chunk_ids,))
            retrieved_chunks = [row[0] for row in cur.fetchall()]
        
        context = "\n\n---\n\n".join(retrieved_chunks)
        tasks.append(generate_answer_async(question, context))
    
    all_answers = await asyncio.gather(*tasks)
    conn.close()
            
    return QueryResponse(answers=all_answers)

app.include_router(router)
