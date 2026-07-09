import os
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Global cache to avoid reloading RAG on every request
_rag_cache = {"chain": None, "retriever": None}

def load_rag():
    """Load RAG chain and retriever. Uses cache to avoid reloading."""
    # Return cached version if available
    if _rag_cache["chain"] is not None and _rag_cache["retriever"] is not None:
        return _rag_cache["chain"], _rag_cache["retriever"]
    
    print("[INFO] Loading embeddings and vector store...")
    embedding_model = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    print("[INFO] Loading LLM...")
    llm = ChatGroq(
            model="llama-3.1-8b-instant", # الموديل الأكبر والأقوى في اللغة العربية
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
    
    template = """
You are an advanced AI assistant specialized in analyzing Egyptian road damage reports.
Your task is to provide a highly professional, fluent, and natural Arabic response based ONLY on the provided reports.

Guidelines for your response:
1. Write in clear, eloquent, and grammatically correct Arabic (لغة عربية فصحى بليغة وسلسة).
2. Avoid literal translations (ترجمة حرفية) or broken phrasing. Make it sound professional.
3. If the answer cannot be found in the reports, strictly reply with: "لم يتم العثور على تفاصيل مطابقة في التقارير الحالية."
4. Do not mention "the context" or "the provided text" in your Arabic answer. Just give the facts naturally.

Reports:
{context}

Question: {question}

Answer in fluent Arabic:
"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Cache for future requests
    _rag_cache["chain"] = chain
    _rag_cache["retriever"] = retriever
    
    print("[INFO] ✓ RAG chain loaded and cached!")
    return chain, retriever
def ask(question: str):
    chain, retriever = load_rag()
    answer = chain.invoke(question)
    sources = [doc.metadata.get("source", "unknown") for doc in retriever.invoke(question)]
    
    with open("answer.txt", "w", encoding="utf-8") as f:
        f.write(f"Question: {question}\n\n")
        f.write(f"Answer:\n{answer}\n\n")
        f.write("Sources:\n")
        for s in sources:
            f.write(f"  - {s}\n")
            
    print("\n[✓] Done! Saved to answer.txt")
    
    return answer

if __name__ == "__main__":
    question = input("Ask your question: ")
    ask(question)