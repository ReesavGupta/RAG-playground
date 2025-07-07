import os
from ingestion.pdf_loader import load_document
from ingestion.chunker import chunk_text
from backend.embeddings import embed_and_store

# Paths to your legal documents (absolute paths)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATHS = [
    os.path.join(BASE_DIR, 'docs', 'income-tax-act-1961-as-amended-by-finance-act-2025.pdf'),
    os.path.join(BASE_DIR, 'docs', 'Circular-No-250-2025.pdf')
]

# To add more legal documents (GST, court judgments, property law):
# 1. Place your PDF or Word files in the 'docs' folder.
# 2. Add their paths to the PDF_PATHS list below.
# Alternatively, use the web UI to upload and ingest documents directly.

def ingest_files(file_paths):
    all_chunks = []
    for path in file_paths:
        print(f'Loading {path}...')
        pages = load_document(path)
        for page in pages:
            chunks = chunk_text(page.page_content)
            all_chunks.extend(chunks)
    print(f'Total chunks to embed: {len(all_chunks)}')
    embed_and_store(all_chunks)
    print('Ingestion and indexing complete!')

if __name__ == '__main__':
    ingest_files(PDF_PATHS) 