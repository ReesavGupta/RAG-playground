from langchain_chroma import Chroma
from langchain_huggingface.embeddings  import  HuggingFaceEmbeddings

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def store_embedding(documents , persist_path = "index/"):
    embeddings =  HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_path)
    print("created vector db with all the indexes")
    return db

def load_vector_store(persist_path = "index/"):
    
    embeddings =  HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Load existing Chroma database
    db = Chroma(
        persist_directory=persist_path,
        embedding_function=embeddings,
    )
    
    print(f"Loaded Chroma database from: {persist_path}")
    return db