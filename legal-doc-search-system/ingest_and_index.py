import os
from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_text
from backend.embeddings import embed_and_store

# Paths to your legal documents (absolute paths)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATHS = [
    os.path.join(BASE_DIR, 'docs', 'income-tax-act-1961-as-amended-by-finance-act-2025.pdf'),
    os.path.join(BASE_DIR, 'docs', 'Circular-No-250-2025.pdf')
]

def main():
    all_chunks = []
    for path in PDF_PATHS:
        print(f'Loading {path}...')
        pages = load_pdf(path)
        for page in pages:
            # page.page_content is the text content
            chunks = chunk_text(page.page_content)
            all_chunks.extend(chunks)
    print(f'Total chunks to embed: {len(all_chunks)}')
    embed_and_store(all_chunks)
    print('Ingestion and indexing complete!')

if __name__ == '__main__':
    main() 