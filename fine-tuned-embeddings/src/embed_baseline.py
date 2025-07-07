from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings

def generate_baseline_embeddings(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return [embedding_model.embed_documents(chunk) for chunk in chunks]
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # return [model.encode(chunk) for chunk in chunks]

