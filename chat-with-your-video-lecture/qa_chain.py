from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", k=4)
    chain = RetrievalQA.from_chain_type(
        llm=OllamaLLM(model="llama3:8b",temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return chain

