import streamlit as st
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests
import json

# ---- Load and Prepare Data ----
@st.cache_resource(show_spinner=True)
def load_and_prepare():
    excel_dir = "excel_files/"
    docs = []
    metadata = []

    for fname in os.listdir(excel_dir):
        if fname.endswith('.xlsx'):
            df = pd.read_excel(os.path.join(excel_dir, fname))
            table_name = os.path.splitext(fname)[0]

            for idx, row in df.iterrows():
                field_name = str(row.get('Field Name', '')).strip()
                redshift_element = str(row.get('Redshift Element Name', '')).strip()
                data_length = str(row.get('Data Length', '')).strip()
                description = str(row.get('Description', '')).strip()
                programmer_notes = str(row.get('Programmer Notes', '')).strip()
                join_criteria = str(row.get('Joining Criteria SQL', '')).strip()

                # Create multiple embeddings by column to enhance retrieval granularity
                if field_name:
                    docs.append(f"Field Name: {field_name}\nTable: {table_name}")
                    metadata.append({'table': table_name, 'column': 'Field Name', 'field': field_name})
                if redshift_element:
                    docs.append(f"Redshift Element Name: {redshift_element}\nTable: {table_name}\nField Name: {field_name}")
                    metadata.append({'table': table_name, 'column': 'Redshift Element Name', 'field': field_name})
                if data_length:
                    docs.append(f"Data Length: {data_length}\nField Name: {field_name}\nTable: {table_name}")
                    metadata.append({'table': table_name, 'column': 'Data Length', 'field': field_name})
                if description:
                    docs.append(f"Description: {description}\nField Name: {field_name}\nTable: {table_name}")
                    metadata.append({'table': table_name, 'column': 'Description', 'field': field_name})
                if programmer_notes:
                    docs.append(f"Programmer Notes: {programmer_notes}\nField Name: {field_name}\nTable: {table_name}")
                    metadata.append({'table': table_name, 'column': 'Programmer Notes', 'field': field_name})
                if join_criteria:
                    docs.append(f"Joining Criteria SQL: {join_criteria}\nField Name: {field_name}\nTable: {table_name}")
                    metadata.append({'table': table_name, 'column': 'Joining Criteria SQL', 'field': field_name})

    st.write(f"✅ Indexed {len(docs)} document chunks across all Excel files.")

    # Embed all docs
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(docs, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return model, index, docs, metadata

# ---- Retrieve Top-k Docs ----
def retrieve_relevant_docs(query, model, index, docs, metadata, k=5):
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb), k)
    results = [(docs[i], metadata[i]) for i in I[0]]

    st.write("🔍 Top matched context:")
    for doc, meta in results:
        st.code(f"{doc}\n\n🔖 Meta: {meta}")

    return results

# ---- Call Ollama ----
def call_ollama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama2", "prompt": prompt},
            stream=True
        )

        answer_chunks = []
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    chunk = data.get("response", "")
                    answer_chunks.append(chunk)
                except Exception:
                    pass  # skip bad JSON

        return "".join(answer_chunks).strip()

    except Exception as e:
        return f"❌ Error contacting Ollama server: {e}"

# ---- Answer Query --------                                
def answer_query(user_query, model, index, docs, metadata):
    top_matches = retrieve_relevant_docs(user_query, model, index, docs, metadata)
    context_docs = [doc for doc, _ in top_matches]

    context_str = "\n\n".join(context_docs)

    prompt_template = f"""You are an assistant who answers developer queries about our Redshift schema.
Here is relevant documentation:

{context_str}

User question:
{user_query}

Based only on the documentation above, answer the question clearly."""

    answer = call_ollama(prompt_template)

    with st.expander("📄 Show Retrieved Context"):
        for i, (doc, meta) in enumerate(top_matches):
            st.markdown(f"**Context #{i+1}** — `{meta['table']}` | `{meta['column']}` for `{meta['field']}`")
            st.code(doc.strip())

    return answer.strip()

# ---- Streamlit UI ----
st.set_page_config(page_title="Redshift RAG Chatbot", layout="wide")
st.title("🧑‍💻 Redshift Documentation Assistant")

with st.spinner('⏳ Loading documents and models...'):
    embed_model, faiss_index_obj, docs_list, metadata_list = load_and_prepare()

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

user_input = st.text_input(
    "Ask a question about your Redshift columns or table joins:",
    placeholder="E.g., What does user_id mean? How do I join claims and members?"
)

# Handle question submission
if st.button("Ask") or (user_input and not st.session_state.get("already_asked")):
    if user_input.strip():
        with st.spinner("🧠 Thinking..."):
            answer_output = answer_query(
                user_input.strip(),
                embed_model,
                faiss_index_obj,
                docs_list,
                metadata_list
            )
            st.session_state.chat_history.append({
                "q": user_input.strip(),
                "a": answer_output
            })
            st.session_state["already_asked"] = True

# Display chat history
for pair in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(pair["q"])
    with st.chat_message("assistant"):
        st.markdown(pair["a"])

