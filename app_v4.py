import streamlit as st
import os
import shutil
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests
import json
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# ---- SQLAlchemy ORM Setup ----
Base = declarative_base()
DB_PATH = "sqlite:///chat_history.db"

class ChatHistory(Base):
    __tablename__ = 'chat_history'
    id = Column(Integer, primary_key=True)
    question = Column(Text)
    answer = Column(Text)
    context = Column(Text)
    timestamp = Column(DateTime)
    thread_id = Column(String)  # For threading support

engine = create_engine(DB_PATH)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

def save_chat(question, answer, context, thread_id):
    db = SessionLocal()
    entry = ChatHistory(
        question=question,
        answer=answer,
        context=context,
        timestamp=datetime.now(),
        thread_id=thread_id
    )
    db.add(entry)
    db.commit()
    db.close()

def get_chat_history(thread_id=None):
    db = SessionLocal()
    if thread_id:
        entries = db.query(ChatHistory).filter(ChatHistory.thread_id == thread_id).order_by(ChatHistory.timestamp).all()
    else:
        entries = db.query(ChatHistory).order_by(ChatHistory.timestamp).all()
    db.close()
    return entries

def clear_chat_history():
    db = SessionLocal()
    db.query(ChatHistory).delete()
    db.commit()
    db.close()

def download_chat_history():
    data_tuples = [(c.timestamp.strftime("%Y-%m-%d %H:%M:%S"), c.question, c.answer, c.context, c.thread_id) for c in get_chat_history()]
    df = pd.DataFrame(data_tuples, columns=["Timestamp", "Question", "Answer", "Context", "Thread ID"])
    return df.to_csv(index=False).encode("utf-8")

# ---- Load and Prepare Data ----
@st.cache_resource(show_spinner=True)
def load_and_prepare():
    excel_dir = "excel_files/"
    docs = []
    metadata = []

    if not os.path.exists(excel_dir):
        os.makedirs(excel_dir)

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

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(docs, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return model, index, docs, metadata

# ---- Query Handling ----
def retrieve_relevant_docs(query, model, index, docs, metadata, k=5):
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb), k)
    return [(docs[i], metadata[i]) for i in I[0]]

def call_ollama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False}
        )
        data = response.json()
        print("\n🟩 Ollama full JSON response:\n", json.dumps(data, indent=2))
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"❌ HTTP error: {e}")
        return "❌ Unable to contact Ollama server."
    except json.JSONDecodeError:
        return "❌ Received invalid JSON from Ollama."
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return f"❌ Error: {e}"

def answer_query(user_query, model, index, docs, metadata, thread_id):
    top_matches = retrieve_relevant_docs(user_query, model, index, docs, metadata)
    context_docs = [doc for doc, _ in top_matches]
    context_str = "\n\n".join(context_docs)

    prompt = f"""You are an assistant who answers Redshift-related developer queries. 
Here is relevant documentation:

{context_str}

User question:
{user_query}

Answer in one short, simple, and conversational sentence — no headers or formatting."""

    answer = call_ollama(prompt)
    save_chat(user_query, answer, context_str, thread_id)
    return answer.strip(), top_matches

# ---- Streamlit UI ----
st.set_page_config(page_title="Redshift RAG Chatbot", layout="wide")
st.title("🧑‍💻 Redshift Documentation Assistant")

uploaded_files = st.file_uploader("Upload Redshift Excel files", type=["xlsx"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join("excel_files", file.name), "wb") as f:
            shutil.copyfileobj(file, f)
    st.success("📥 Files uploaded. Please refresh the page.")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.user_has_asked = False

with st.spinner("Loading and embedding documents..."):
    model, index, docs_list, metadata_list = load_and_prepare()

thread_id = st.text_input("🔗 Enter or create a thread ID to group questions:", value="default")

user_input = st.text_input(
    "Ask a question about your Redshift columns or table joins:",
    placeholder="E.g., What does user_id mean? How do I join claims and members?"
)

if st.button("Ask"):
    if user_input.strip():
        with st.spinner("🧠 Thinking..."):
            answer, matches = answer_query(user_input.strip(), model, index, docs_list, metadata_list, thread_id)
            st.session_state.chat_history.append({"q": user_input.strip(), "a": answer, "context": matches})
            st.session_state.user_has_asked = True

# ---- Show Chat History ----
filter_thread = st.text_input("🔍 Filter chat history by thread ID:")
entries = get_chat_history(thread_id=filter_thread or None)

for idx, chat in enumerate(st.session_state.chat_history[::-1]):
    with st.chat_message("user"):
        st.markdown(chat['q'])
    with st.chat_message("assistant"):
        st.markdown("#### 💬 Answer:")
        st.success(chat['a'] if chat['a'] else "Sorry, no answer generated.")
        st.code(chat['a'], language='markdown')
        st.button("📋 Copy Answer", key=f"copy_{idx}")
        with st.expander("📄 Show Retrieved Context"):
            for i, (doc, meta) in enumerate(chat["context"]):
                st.markdown(f"**Context {i+1}**: `{meta['table']}` – `{meta['column']}` for `{meta['field']}`")
                st.code(doc.strip())

# ---- Tools ----
st.download_button("📥 Download Chat History", data=download_chat_history(), file_name="chat_history.csv")
if st.button("🗑️ Clear All History"):
    clear_chat_history()
    st.success("Chat history cleared. Please refresh.")
