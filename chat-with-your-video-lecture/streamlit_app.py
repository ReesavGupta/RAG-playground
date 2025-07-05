import streamlit as st
from vector_stoer import load_vector_store
from qa_chain import build_qa_chain
from utils import summarize_sources

st.title("🎓 Chat with Your Lecture")
query = st.text_input("Ask a question about the lecture")

if query:
    db = load_vector_store()
    qa_chain = build_qa_chain(db)
    result = qa_chain({"query": query})
    st.write("🧠 Answer:", result["result"])
    st.markdown("⏱️ Sources:")
    st.text(summarize_sources(result["source_documents"]))
