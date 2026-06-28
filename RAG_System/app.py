from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from query import load_rag

app = FastAPI(title="Road Guard RAG API")

chain = None
retriever = None

@app.on_event("startup")
def startup_event():
    global chain, retriever
    print("Loading RAG chain...")
    chain, retriever = load_rag()
    print("RAG chain ready.")

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        answer = chain.invoke(request.question)
        docs = retriever.invoke(request.question)
        sources = list(set([doc.metadata.get("source", "unknown") for doc in docs]))
        return AnswerResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Road Guard RAG is running."}