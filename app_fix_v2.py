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

class ChatSession(Base):
    __tablename__ = 'chat_session'
    id = Column(Integer, primary_key=True)
    name = Column(String, default="Untitled Chat")
    created_at = Column(DateTime, default=datetime.now)

class ChatHistory(Base):
    __tablename__ = 'chat_history'
    id = Column(Integer, primary_key=True)
    question = Column(Text)
    answer = Column(Text)
    context = Column(Text)
    timestamp = Column(DateTime)
    feedback = Column(String)
    session_id = Column(Integer)

engine = create_engine(DB_PATH)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# ---- DB Functions ----
def save_chat(question, answer, context, session_id):
    db = SessionLocal()
    entry = ChatHistory(
        question=question,
        answer=answer,
        context=context,
        timestamp=datetime.now(),
        feedback=None,
        session_id=session_id,
    )
    db.add(entry)
    db.commit()
    db.close()

def get_chat_history(session_id):
    db = SessionLocal()
    entries = db.query(ChatHistory).filter_by(session_id=session_id).order_by(ChatHistory.timestamp).all()
    db.close()
    return entries

def clear_chat_history():
    db = SessionLocal()
    db.query(ChatHistory).delete()
    db.commit()
    db.close()

def download_chat_history():
    data_tuples = [(c.timestamp.strftime("%Y-%m-%d %H:%M:%S"), c.question, c.answer, c.context, c.feedback) for c in get_chat_history(st.session_state.selected_session_id)]
    df = pd.DataFrame(data_tuples, columns=["Timestamp", "Question", "Answer", "Context", "Feedback"])
    return df.to_csv(index=False).encode("utf-8")

# ---- Load and Prepare Data ----
@st.cache_resource(show_spinner=True)
def load_and_prepare():
    excel_dir = "excel_files/"
    row_docs = []
    field_docs = []
    row_metadata = []
    field_metadata = []

    if not os.path.exists(excel_dir):
        os.makedirs(excel_dir)

    for fname in os.listdir(excel_dir):
        if fname.endswith('.xlsx'):
            df = pd.read_excel(os.path.join(excel_dir, fname), engine='openpyxl')
            table_name = os.path.splitext(fname)[0]

            for idx, row in df.iterrows():
                row_summary = []
                for col in df.columns:
                    val = str(row.get(col, '')).strip()
                    if val:
                        row_summary.append(f"{col.strip()}: {val}")
                        field_docs.append(f"{col.strip()}: {val}\nTable: {table_name}")
                        field_metadata.append({'table': table_name, 'column': col, 'row': idx})
                if row_summary:
                    doc_text = "\n".join(row_summary) + f"\nTable: {table_name}, Row: {idx}"
                    row_docs.append(doc_text)
                    row_metadata.append({'table': table_name, 'row_index': idx})

    model = SentenceTransformer('all-MiniLM-L6-v2')
    row_embeddings = model.encode(row_docs, show_progress_bar=True)
    field_embeddings = model.encode(field_docs, show_progress_bar=True)

    dim = row_embeddings.shape[1]
    row_index = faiss.IndexFlatL2(dim)
    row_index.add(np.array(row_embeddings))

    field_index = faiss.IndexFlatL2(dim)
    field_index.add(np.array(field_embeddings))

    return model, row_index, row_docs, row_metadata, field_index, field_docs, field_metadata

# ---- Query Handling ----
def retrieve_relevant_docs(query, model, row_index, row_docs, row_metadata, field_index, field_docs, field_metadata, k=5):
    query_emb = model.encode([query])
    Dr, Ir = row_index.search(np.array(query_emb), k)
    Df, If = field_index.search(np.array(query_emb), k)

    row_results = [(row_docs[i], row_metadata[i]) for i, dist in zip(Ir[0], Dr[0]) if dist < 1.5]
    field_results = [(field_docs[i], field_metadata[i]) for i, dist in zip(If[0], Df[0]) if dist < 1.5]

    combined = {doc: meta for doc, meta in row_results + field_results}
    return list(combined.items())

def call_ollama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False}
        )
        data = response.json()
        print("\n🟩 Ollama full JSON response:\n", json.dumps(data, indent=2))
        return data.get("response", "").strip()
    except Exception as e:
        return f"❌ Error contacting Ollama server: {e}"

def answer_query(user_query, model, row_index, row_docs, row_metadata, field_index, field_docs, field_metadata):
    top_matches = retrieve_relevant_docs(user_query, model, row_index, row_docs, row_metadata, field_index, field_docs, field_metadata)
    context_docs = [doc for doc, _ in top_matches]
    context_str = "\n---\n".join(context_docs)

    prompt = f"""You are a Redshift assistant. Use the following context to explain the answer in detail if necessary. If the user requests SQL, generate the SQL snippet.

Context:
{context_str}

User query:
{user_query}

Reply clearly. Include SQL if asked. Keep responses conversational but thorough."""
    answer = call_ollama(prompt)
    save_chat(user_query, answer, context_str, st.session_state.selected_session_id)
    return answer.strip(), top_matches

# ---- Streamlit UI ----
st.set_page_config(page_title="Redshift RAG Chatbot", layout="wide")
st.title("🧑‍💻 Redshift Documentation Assistant")

# Sidebar for chat session management
st.sidebar.title("💬 Chat Sessions")
with st.sidebar.form("new_session"):
    new_chat_name = st.text_input("New chat name")
    if st.form_submit_button("Start New Chat") and new_chat_name:
        db = SessionLocal()
        new_session = ChatSession(name=new_chat_name)
        db.add(new_session)
        db.commit()
        st.session_state.selected_session_id = new_session.id
        db.close()
        st.rerun()

db = SessionLocal()
sessions = db.query(ChatSession).order_by(ChatSession.created_at.desc()).all()
db.close()
session_options = {s.name: s.id for s in sessions}

if 'selected_session_id' not in st.session_state:
    if sessions:
        st.session_state.selected_session_id = sessions[0].id
    else:
        # Create a default session if none exist
        db = SessionLocal()
        default_session = ChatSession(name="Default Chat")
        db.add(default_session)
        db.commit()
        st.session_state.selected_session_id = default_session.id
        db.close()

selected_chat_name = None
if session_options:
    selected_chat_name = st.sidebar.selectbox("Select Chat", list(session_options.keys()), index=0)
    st.session_state.selected_session_id = session_options[selected_chat_name]

uploaded_files = st.file_uploader("Upload Redshift Excel files", type=["xlsx"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join("excel_files", file.name), "wb") as f:
            shutil.copyfileobj(file, f)
    st.success("📥 Files uploaded. Please refresh the page.")

with st.spinner("Loading and embedding documents..."):
    model, row_index, row_docs_list, row_metadata_list, field_index, field_docs_list, field_metadata_list = load_and_prepare()

if 'chat_history' not in st.session_state or st.session_state.chat_history is None:
    st.session_state.chat_history = get_chat_history(st.session_state.selected_session_id)

user_input = st.text_input(
    f"Ask a question about your Redshift columns or table joins (Session: {selected_chat_name}):",
    placeholder="E.g., What does user_id mean? How do I join claims and members?",
    key="user_input"
)

if st.button("Ask"):
    if user_input.strip():
        with st.spinner("🧠 Thinking..."):
            answer, matches = answer_query(
                user_input.strip(),
                model,
                row_index, row_docs_list, row_metadata_list,
                field_index, field_docs_list, field_metadata_list
            )
            st.session_state.chat_history.append(ChatHistory(
                question=user_input.strip(),
                answer=answer,
                context="\n\n".join([doc for doc, _ in matches]),
                timestamp=datetime.now(),
                session_id=st.session_state.selected_session_id
            ))

# ---- Show Chat History ----
for entry in st.session_state.chat_history[::-1]:
    with st.chat_message("user"):
        st.markdown(entry.question)
    with st.chat_message("assistant"):
        st.markdown("#### 💬 Answer:")
        st.success(entry.answer if entry.answer else "Sorry, no answer generated.")
        with st.expander("📄 Show Retrieved Context"):
            for i, doc in enumerate(entry.context.strip().split("\n\n")):
                st.markdown(f"**Context {i+1}**:")
                st.code(doc)

# ---- Tools ----
st.download_button("📥 Download Chat History", data=download_chat_history(), file_name="chat_history.csv")
if st.button("🗑️ Clear All History"):
    clear_chat_history()
    st.session_state.chat_history = []
    st.success("Chat history cleared. Please refresh.")
