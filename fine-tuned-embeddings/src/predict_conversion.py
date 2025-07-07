from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def predict_conversion(query, model_path, persist_dir="chroma_db", k=5):
    embedding_fn = HuggingFaceEmbeddings(model_name=model_path)
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_fn)

    results = vectorstore.similarity_search_with_score(query, k=k)
    converted_count = sum(1 for doc, score in results if doc.metadata['label'] == 1)
    return converted_count / k