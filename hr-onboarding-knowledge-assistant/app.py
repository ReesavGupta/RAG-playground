import streamlit as st
from qa_chain import get_qa_chain

st.set_page_config(page_title="HR Assistant", page_icon="🤖")
st.title("🧑‍💼 HR Onboarding Knowledge Assistant")

qa_chain = get_qa_chain()

query = st.text_input("Ask your HR question:")

if st.button("Get Answer"):
    if query.strip():
        result = qa_chain.invoke({"query": query})
        st.write("### ✅ Answer")
        st.write(result["result"])

        st.write("### 📚 Sources")
        for doc in result["source_documents"]:
            st.write(f"- {doc.metadata.get('source', 'Unknown')}")
