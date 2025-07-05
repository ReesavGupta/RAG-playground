from langchain_chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings

def store_embedding(documents , persist_path = "index/"):
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_path)
    print("created vector db with all the indexes")
    return db

def load_vector_store(persist_path = "index/"):
    embeddings = OpenAIEmbeddings()
    
    # Load existing Chroma database
    db = Chroma(
        persist_directory=persist_path,
        embedding_function=embeddings,
    )
    
    print(f"Loaded Chroma database from: {persist_path}")
    return db