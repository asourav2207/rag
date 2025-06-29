import os
import streamlit as st
from main import load_documents, embed_docs, build_faiss_index, answer_query

EXCEL_DIR = "excel_files"
os.makedirs(EXCEL_DIR, exist_ok=True)

st.set_page_config(page_title="Redshift QA", layout="centered")
st.title("🔍 Ask Redshift Schema Questions")

# Step 1: Upload Excel
uploaded_file = st.file_uploader("Upload Core Certification Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    file_path = os.path.join(EXCEL_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded: {uploaded_file.name}")

    try:
        # Step 2: Load and Build FAISS index
        docs, _ = load_documents(EXCEL_DIR)
        model, embeddings = embed_docs(docs)
        index = build_faiss_index(embeddings)

        # Step 3: Ask a question
        st.markdown("### Ask a Question:")
        query = st.text_input("Your Question")
        if st.button("Submit") and query:
            with st.spinner("Thinking..."):
                answer = answer_query(query, model, index, docs)
                st.markdown("### 📘 Answer")
                # Ensure we coerce to string, in case it's a list or None
                answer_str = str(answer) if answer else "No response."

# Normalize line breaks
                answer_str = answer_str.replace("\\n", "\n")

# Display in a larger scrollable text area
                st.text_area("LLM Response", value=answer_str, height=400, max_chars=None, key="response_box")



    except Exception as e:
        st.error(f"Error: {str(e)}")
