import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from backend.embeddings import embedding_model, vectordb
from backend.similiarity import cosine_search, euclidean_search, mmr_search, hybrid_score, precision_at_k, recall_at_k, diversity_score
from backend.legal_entity_match import legal_entity_match
from sklearn.metrics.pairwise import cosine_similarity

st.title("Indian Legal Document Search System")

uploaded_files = st.file_uploader("Upload legal documents (PDF or Word)", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    os.makedirs("uploaded_docs", exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = os.path.join("uploaded_docs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded {len(uploaded_files)} file(s). They will be indexed when you click 'Ingest Documents'.")

ingest_btn = st.button("Ingest Documents")

if ingest_btn and uploaded_files:
    from ingestion.pdf_loader import load_document
    from ingestion.chunker import chunk_text
    from backend.embeddings import embed_and_store
    file_paths = [os.path.join("uploaded_docs", f.name) for f in uploaded_files]
    all_chunks = []
    for path in file_paths:
        pages = load_document(path)
        for page in pages:
            chunks = chunk_text(page.page_content)
            all_chunks.extend(chunks)
    embed_and_store(all_chunks)
    st.success(f"Ingested and indexed {len(file_paths)} document(s).")

query = st.text_input("Enter your legal query")

search_btn = st.button("Search")

if search_btn and query:
    query_embedding = embedding_model.embed_query(query)
    all_docs = vectordb.similarity_search_by_vector(query_embedding, k=20)
    doc_embeddings = [embedding_model.embed_query(doc.page_content) for doc in all_docs]
    cosine_results = cosine_search(vectordb, query)
    euclidean_indices = euclidean_search(query_embedding, doc_embeddings)
    euclidean_results = [all_docs[i] for i in euclidean_indices]
    mmr_results = mmr_search(vectordb, query)
    hybrid_results = sorted(
        all_docs,
        key=lambda d: hybrid_score(
            cosine_similarity([query_embedding], [embedding_model.embed_query(d.page_content)])[0][0],
            legal_entity_match(query, d)
        ),
        reverse=True
    )[:5]
    st.session_state['last_query'] = query
    st.session_state['last_results'] = {
        'cosine': cosine_results,
        'euclidean': euclidean_results,
        'mmr': mmr_results,
        'hybrid': hybrid_results,
        'all_docs': all_docs,
        'query_embedding': query_embedding,
        'doc_embeddings': doc_embeddings
    }

# Always display results if present in session_state
if 'last_results' in st.session_state:
    results = st.session_state['last_results']
    cosine_results = results['cosine']
    euclidean_results = results['euclidean']
    mmr_results = results['mmr']
    hybrid_results = results['hybrid']
    all_docs = results['all_docs']
    cols = st.columns(4)
    titles = ["Cosine", "Euclidean", "MMR", "Hybrid"]
    all_methods = [cosine_results, euclidean_results, mmr_results, hybrid_results]

    if 'relevance' not in st.session_state:
        st.session_state['relevance'] = {}

    for col, title, results_list in zip(cols, titles, all_methods):
        with col:
            st.header(title)
            for i, doc in enumerate(results_list):
                st.write(doc.page_content[:300])
                key = f"{title}_{i}_relevant"
                checked = st.checkbox("Relevant?", key=key)
                st.session_state['relevance'][key] = checked

    if st.button("Calculate Metrics"):
        relevant_docs = []
        for method, results_list in zip(titles, all_methods):
            for i, doc in enumerate(results_list):
                key = f"{method}_{i}_relevant"
                if st.session_state['relevance'].get(key, False):
                    relevant_docs.append(doc)
        metrics = {}
        for method, results_list in zip(titles, all_methods):
            metrics[method] = {
                'Precision@5': precision_at_k(results_list, relevant_docs, k=5),
                'Recall@5': recall_at_k(results_list, relevant_docs, k=5),
                'Diversity': diversity_score(results_list)
            }
        st.subheader("Performance Metrics (User Marked)")
        st.table([{**{'Method': m}, **metrics[m]} for m in metrics])