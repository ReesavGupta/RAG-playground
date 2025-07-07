from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document


def build_vectorstore(df, model_path, persist_dir="chroma_db"):
    embedding_fn = HuggingFaceEmbeddings(model_name=model_path)
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_fn)

    documents = []
    for _, row in df.iterrows():
        for chunk in row['chunks']:
            documents.append(Document(page_content=chunk, metadata={"label": row['label']}))

    vectorstore.add_documents(documents)
    return vectorstore