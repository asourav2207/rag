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
import base64

# ---- GitHub Integration Config ----
# Store your GitHub token in Streamlit secrets or as an environment variable
GITHUB_TOKEN = st.secrets.get("github_token", os.environ.get("GITHUB_TOKEN", ""))
GITHUB_REPO = st.secrets.get("github_repo", "asourav2207/rag")  # e.g., "adityasourav/redshift-excels"
GITHUB_BRANCH = st.secrets.get("github_branch", "main")
GITHUB_EXCEL_PATH = st.secrets.get("github_excel_path", "excel_files")  # folder in repo

def github_headers():
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

def list_github_excels():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_EXCEL_PATH}?ref={GITHUB_BRANCH}"
    try:
        r = requests.get(url, headers=github_headers(), timeout=10)
        r.raise_for_status()
        files = r.json()
        return [f for f in files if f["name"].endswith(".xlsx")]
    except Exception as e:
        st.error(f"Error listing Excel files from GitHub: {e}")
        return []

def download_excel_from_github(file_info):
    try:
        r = requests.get(file_info["download_url"], timeout=10)
        r.raise_for_status()
        return pd.read_excel(r.content, engine="openpyxl")
    except Exception as e:
        st.error(f"Error downloading {file_info['name']} from GitHub: {e}")
        return None

def upload_excel_to_github(file, filename):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_EXCEL_PATH}/{filename}"
    try:
        # Get SHA if file exists (for update)
        get_resp = requests.get(url, headers=github_headers(), timeout=10)
        sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None

        file.seek(0)
        content = base64.b64encode(file.read()).decode()
        data = {
            "message": f"Add {filename}",
            "content": content,
            "branch": GITHUB_BRANCH,
        }
        if sha:
            data["sha"] = sha
        put_resp = requests.put(url, headers=github_headers(), data=json.dumps(data), timeout=15)
        if put_resp.status_code in (200, 201):
            return True
        else:
            st.error(f"GitHub upload failed for {filename}: {put_resp.json().get('message', put_resp.text)}")
            return False
    except Exception as e:
        st.error(f"Error uploading {filename} to GitHub: {e}")
        return False

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
    def process_row(row, df_columns, table_name, idx, row_docs, field_docs, row_metadata, field_metadata):
        row_summary = []
        for col in df_columns:
            val = str(row.get(col, '')).strip()
            if val:
                row_summary.append(f"{col.strip()}: {val}")
                field_docs.append(f"{col.strip()}: {val}\nTable: {table_name}")
                field_metadata.append({'table': table_name, 'column': col, 'row': idx})
        if row_summary:
            doc_text = "\n".join(row_summary) + f"\nTable: {table_name}, Row: {idx}"
            row_docs.append(doc_text)
            row_metadata.append({'table': table_name, 'row_index': idx})

    def process_excel_file(file_info, row_docs, field_docs, row_metadata, field_metadata):
        df = download_excel_from_github(file_info)
        if df is None:
            return
        table_name = os.path.splitext(file_info["name"])[0]
        for idx, row in df.iterrows():
            process_row(row, df.columns, table_name, idx, row_docs, field_docs, row_metadata, field_metadata)

    row_docs = []
    field_docs = []
    row_metadata = []
    field_metadata = []

    # --- Load Excel files from GitHub ---
    excel_files = list_github_excels()
    if not excel_files:
        st.warning("No Excel files found in the GitHub repository.")
    for file_info in excel_files:
        process_excel_file(file_info, row_docs, field_docs, row_metadata, field_metadata)

    if not row_docs or not field_docs:
        st.warning("No data found in the Excel files.")
        # Return empty objects to avoid downstream errors
        model = SentenceTransformer('all-MiniLM-L6-v2')
        dim = 384
        row_index = faiss.IndexFlatL2(dim)
        field_index = faiss.IndexFlatL2(dim)
        return model, row_index, [], [], field_index, [], []

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
    if not row_docs or not field_docs:
        return []
    query_emb = model.encode([query])
    Dr, Ir = row_index.search(np.array(query_emb), k)
    Df, If = field_index.search(np.array(query_emb), k)

    row_results = [(row_docs[i], row_metadata[i]) for i, dist in zip(Ir[0], Dr[0]) if dist < 1.5]
    field_results = [(field_docs[i], field_metadata[i]) for i, dist in zip(If[0], Df[0]) if dist < 1.5]

    combined = dict(row_results + field_results)
    return list(combined.items())

def call_ollama(prompt):
    try:
        # Use 127.0.0.1 instead of localhost for better compatibility
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False},
            timeout=60  # Increased timeout
        )
        response.raise_for_status()
        data = response.json()
        print("\n🟩 Ollama full JSON response:\n", json.dumps(data, indent=2))
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error contacting Ollama server: {e}\nDetails: {getattr(e, 'response', None)}")
        print(f"❌ Error contacting Ollama server: {e}\nDetails: {getattr(e, 'response', None)}")
        return f"❌ Error contacting Ollama server: {e}"
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        return f"❌ Unexpected error: {e}"

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
        st.session_state.chat_history = []
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

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = get_chat_history(st.session_state.selected_session_id)
elif st.session_state.get("last_session_id") != st.session_state.selected_session_id:
    st.session_state.chat_history = get_chat_history(st.session_state.selected_session_id)
st.session_state.last_session_id = st.session_state.selected_session_id

# ---- Upload Excel files to GitHub ----
uploaded_files = st.file_uploader("Upload Redshift Excel files (uploaded to GitHub repo)", type=["xlsx"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        file.seek(0)
        if upload_excel_to_github(file, file.name):
            st.success(f"✅ {file.name} uploaded to GitHub.")
        else:
            st.error(f"❌ Failed to upload {file.name} to GitHub.")
    st.info("Please refresh the page after upload.")

with st.spinner("Loading and embedding documents..."):
    model, row_index, row_docs_list, row_metadata_list, field_index, field_docs_list, field_metadata_list = load_and_prepare()

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
