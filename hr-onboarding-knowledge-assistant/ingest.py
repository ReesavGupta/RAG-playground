import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import EMBEDDING_MODEL, VECTOR_DB_PATH

def load_documents():
    files = [f for f in os.listdir('data') if f.endswith((".pdf", ".docx"))]
    docs = []
    for f in files:
        path = os.path.join("data", f)
        loader = PyPDFLoader(path) if f.endswith(".pdf") else Docx2txtLoader(path)
        docs.extend(loader.load())
    return docs

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
        keep_separator='end',
        is_separator_regex=False
    )
    return splitter.split_documents(documents=documents)

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=VECTOR_DB_PATH)
    print("created vector db")

if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    create_vector_store(chunks)
