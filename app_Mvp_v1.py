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
    feedback = Column(String)

engine = create_engine(DB_PATH)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# ---- DB Functions ----
def save_chat(question, answer, context):
    db = SessionLocal()
    entry = ChatHistory(
        question=question,
        answer=answer,
        context=context,
        timestamp=datetime.now(),
        feedback=None,
    )
    db.add(entry)
    db.commit()
    db.close()

def get_chat_history():
    db = SessionLocal()
    entries = db.query(ChatHistory).order_by(ChatHistory.timestamp).all()
    db.close()
    return entries

def clear_chat_history():
    db = SessionLocal()
    db.query(ChatHistory).delete()
    db.commit()
    db.close()

def download_chat_history():
    data_tuples = [(c.timestamp.strftime("%Y-%m-%d %H:%M:%S"), c.question, c.answer, c.context, c.feedback) for c in get_chat_history()]
    df = pd.DataFrame(data_tuples, columns=["Timestamp", "Question", "Answer", "Context", "Feedback"])
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
                for col_name in df.columns:
                    value = str(row.get(col_name, '')).strip()
                    if value:
                        docs.append(f"{col_name}: {value}\nTable: {table_name}")
                        metadata.append({'table': table_name, 'column': col_name, 'field': str(row.get('Field Name', '')).strip()})

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
    except Exception as e:
        return f"❌ Error contacting Ollama server: {e}"

def answer_query(user_query, model, index, docs, metadata):
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
    save_chat(user_query, answer, context_str)
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

with st.spinner("Loading and embedding documents..."):
    model, index, docs_list, metadata_list = load_and_prepare()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = get_chat_history()

user_input = st.text_input(
    "Ask a question about your Redshift columns or table joins:",
    placeholder="E.g., What does user_id mean? How do I join claims and members?",
    key="user_input"
)

if st.button("Ask"):
    if user_input.strip():
        with st.spinner("🧠 Thinking..."):
            answer, matches = answer_query(user_input.strip(), model, index, docs_list, metadata_list)
            st.session_state.chat_history.append(ChatHistory(
                question=user_input.strip(),
                answer=answer,
                context="\n\n".join([doc for doc, _ in matches]),
                timestamp=datetime.now()
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
