from langchain_community.document_loaders import PyPDFLoader

def load_pdf(path_to_pdf: str):
    loader = PyPDFLoader(path_to_pdf)
    return loader.load()
