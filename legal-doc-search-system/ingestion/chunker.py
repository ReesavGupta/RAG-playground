from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(texts: str, chunk_size = 800, overlap =100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.create_documents(texts=[texts])
