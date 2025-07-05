from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config import VECTOR_DB_PATH, EMBEDDING_MODEL, OLLAMA_MODEL

def get_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.1, num_predict=512)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
            You are an answering agent. Use the following policy context to answer the employee question truthfully.

            Context:
            {context}

            Question: {question}
            Answer:
        """
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain
