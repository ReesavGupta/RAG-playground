from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

def load_document(path: str):
    if path.lower().endswith('.pdf'):
        loader = PyPDFLoader(path)
        return loader.load()
    elif path.lower().endswith('.docx'):
        loader = Docx2txtLoader(path)
        return loader.load()
    else:
        raise ValueError('Unsupported file type: ' + path)
