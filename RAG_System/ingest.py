import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings # <--- عدلنا الاستيراد هنا
from langchain_community.vectorstores import Chroma

load_dotenv()

def ingest_reports():
    # 1. تصفير ومسح الكوليكشن القديم تماماً عشان نضمن عدم تداخل الكاش
    if os.path.exists("./chroma_db"):
        print("Cleaning up old database files...")
        import shutil
        shutil.rmtree("./chroma_db")

    print("Loading reports...")
    loader = DirectoryLoader(
        "reports/",
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"Total reports loaded: {len(documents)}")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, # زودنا الـ chunk سنة عشان يستوعب التقرير العربي كامل في قطعة واحدة
        chunk_overlap=50,
        separators=["\n\n", "\n", "،", ". ", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    print("Creating embeddings via HF API and saving to ChromaDB...")
    # <--- ربطنا الـ Ingestion بنفس سيرفر الـ API بتاع الـ app.py بالظبط
    embedding_model = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HF_TOKEN"), 
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )

    print("Successfully saved to ChromaDB!")
    print(f"Total vectors stored: {vectorstore._collection.count()}")

if __name__ == "__main__":
    ingest_reports()