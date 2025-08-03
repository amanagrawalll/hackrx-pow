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
    documents: str

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# --- FastAPI App & Router ---
app = FastAPI(title="Production RAG Service")
router = APIRouter(prefix="/api/v1")

# --- Database Functions ---
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

# --- Core Logic with Improved Transaction Handling ---
def process_and_index_document(document_url: str, google_api_key: str):
    """The background task for indexing, now with robust transaction handling."""
    genai.configure(api_key=google_api_key)
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cur:
            print(f"Starting indexing for: {document_url}")
            # Check if URL is already processed
            cur.execute("SELECT id, processed FROM documents WHERE url = %s", (document_url,))
            result = cur.fetchone()
            
            if result and result[1]:
                print(f"Document {document_url} already processed.")
                return

            if not result:
                cur.execute("INSERT INTO documents (url) VALUES (%s) RETURNING id", (document_url,))
                doc_id = cur.fetchone()[0]
            else:
                doc_id = result[0]
            
            print(f"Document ID is {doc_id}. Processing PDF...")
            response = requests.get(document_url)
            response.raise_for_status()
            with BytesIO(response.content) as pdf_file:
                reader = PdfReader(pdf_file)
                text = "".join(page.extract_text() or "" for page in reader.pages)
            
            chunks = []
            chunk_size, chunk_overlap = 2000, 200
            start = 0
            while start < len(text):
                chunks.append(text[start:start + chunk_size])
                start += chunk_size - chunk_overlap
            
            print(f"Document chunked into {len(chunks)} pieces. Storing in database...")
            chunk_ids = []
            for chunk_text in chunks:
                cur.execute("INSERT INTO chunks (document_id, chunk_text) VALUES (%s, %s) RETURNING id", (doc_id, chunk_text))
                chunk_ids.append(cur.fetchone()[0])
            
            print("Creating embeddings...")
            embeddings = genai.embed_content(model='models/embedding-001', content=chunks, task_type="RETRIEVAL_DOCUMENT")['embedding']
            
            print("Upserting vectors to Pinecone...")
            vectors_to_upsert = [{"id": f"{doc_id}:{chunk_ids[i]}", "values": emb} for i, emb in enumerate(embeddings)]
            pinecone_index.upsert(vectors=vectors_to_upsert, namespace=str(doc_id))
            
            print("Marking document as processed in database...")
            cur.execute("UPDATE documents SET processed = TRUE WHERE id = %s", (doc_id,))
            
            # If all steps above succeed, commit the transaction
            print("Committing transaction to database...")
            conn.commit()
            print(f"✅ Successfully indexed and committed document: {document_url}")

    except Exception as e:
        print(f"❌ An error occurred during indexing: {e}. Rolling back transaction.")
        # If any error occurs, roll back all database changes for this run
        conn.rollback()
    finally:
        # Always close the connection
        print("Closing database connection.")
        conn.close()

# The other functions (generate_answer_async, endpoints) do not need to be changed.
# ... (rest of the code from the previous version) ...

async def generate_answer_async(question: str, context: str):
    """Generates a single answer using Gemini Flash."""
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = f"CONTEXT: {context}\n\nQUESTION: {question}\n\nANSWER:"
    response = await model.generate_content_async(prompt)
    return response.text.strip()

@app.on_event("startup")
def on_startup():
    setup_database()

@router.post("/index", status_code=202, tags=["Indexing"])
async def index_document(request: IndexRequest, background_tasks: BackgroundTasks, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header for Google API Key.")
    google_api_key = authorization.split(" ")[1]
    background_tasks.add_task(process_and_index_document, request.documents, google_api_key)
    return {"message": "Document accepted for indexing. Processing happens in the background."}

@router.post("/hackrx/run", response_model=QueryResponse, tags=["Querying"])
async def run_query(request: QueryRequest, authorization: Optional[str] = Header(None)):
    # ... (no changes to this endpoint from the previous version)
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
