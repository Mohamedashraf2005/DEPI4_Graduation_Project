import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import uvicorn
from query import load_rag

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Road Guard RAG API",
    description="RAG API for querying Egyptian road damage reports",
    version="1.0.0"
)

# ============================================================
# THE CORS FIX: Explicitly whitelist the Vercel frontend
# ============================================================
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://raqibroadmonitor.vercel.app",  # Your live Vercel frontend!
]

# Add CORS middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chain = None
retriever = None

@app.on_event("startup")
def startup_event():
    global chain, retriever
    logger.info("[STARTUP] Loading RAG chain...")
    try:
        chain, retriever = load_rag()
        logger.info("[STARTUP] ✓ RAG chain ready!")
    except Exception as e:
        logger.error(f"[STARTUP] ✗ Failed to load RAG: {str(e)}")
        raise

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Query the RAG system with a question about road damages.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        logger.info(f"[ASK] Processing question: {request.question[:50]}...")
        
        # Get answer from chain
        answer = chain.invoke(request.question)
        
        # Get sources
        docs = retriever.invoke(request.question)
        sources = list(set([doc.metadata.get("source", "unknown") for doc in docs]))
        
        logger.info(f"[ASK] ✓ Successfully answered with {len(sources)} sources")
        return AnswerResponse(answer=answer, sources=sources)
        
    except Exception as e:
        logger.error(f"[ASK] ✗ Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/info")
def root():
    """Root endpoint with API info."""
    return {
        "name": "Road Guard RAG API",
        "version": "1.0.0",
        "description": "RAG API for querying Egyptian road damage reports",
        "endpoints": {
            "health": "/health",
            "ask": "/ask (POST)",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    try:
        if chain is None or retriever is None:
            return {
                "status": "loading",
                "message": "RAG chain is still loading...",
                "ready": False
            }
        return {
            "status": "ok",
            "message": "Road Guard RAG is running and ready.",
            "ready": True
        }
    except Exception as e:
        logger.error(f"[HEALTH] Error: {str(e)}")
        return {
            "status": "error",
            "message": f"Health check failed: {str(e)}",
            "ready": False
        }, 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)