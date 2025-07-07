import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from backend.embeddings import embedding_model, vectordb
from backend.similiarity import cosine_search, euclidean_search, mmr_search, hybrid_score
from backend.legal_entity_match import legal_entity_match
from sklearn.metrics.pairwise import cosine_similarity

st.title("Indian Legal Document Search System")
query = st.text_input("Enter your legal query")

if st.button("Search"):
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

    cols = st.columns(4)
    titles = ["Cosine", "Euclidean", "MMR", "Hybrid"]
    all_methods = [cosine_results, euclidean_results, mmr_results, hybrid_results]

    for col, title, results in zip(cols, titles, all_methods):
        with col:
            st.header(title)
            for doc in results:
                st.write(doc.page_content[:300])