from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", k=4)
    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return chain

