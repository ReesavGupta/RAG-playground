from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="embeddings/", embedding_function=embedding_model)

def embed_and_store(docs):
    vectordb.add_documents(docs)
    return vectordb