import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests

def load_documents(excel_folder):
    docs = []
    metadata = []
    for fname in os.listdir(excel_folder):
        if fname.endswith('.xlsx'):
            df = pd.read_excel(os.path.join(excel_folder, fname))
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
                    metadata.append({'table': table_name, 'column': 'Redshift Element Name', 'field': redshift_element})
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

    return docs, metadata

def embed_docs(docs):
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    embeddings = model.encode(docs, show_progress_bar=True)
    return model, np.array(embeddings)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings) 
    return index

def retrieve_relevant_docs(query, model, index, docs, k=5):
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb), k) 
    return [docs[i] for i in I[0]] 

def call_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama2", "prompt": prompt},
        stream=True  # <-- IMPORTANT
    )

    output = ""
    try:
        for line in response.iter_lines():
            if line:
                try:
                    # Each line is a JSON string with a 'response' key
                    part_data = eval(line.decode("utf-8"))  # or use json.loads if safe
                    output += part_data.get("response", "")
                except Exception:
                    continue
    except Exception as e:
        return f"[Error receiving response: {str(e)}]"

    return output.strip()


def answer_query(user_query, model, index, docs):
    context_docs = retrieve_relevant_docs(user_query, model, index, docs)
    context_str = "\n\n".join(context_docs)

    prompt_template = f"""You are an assistant who answers developer queries about our Redshift schema.
Use only the documentation below and respond with a specific, clear answer. Prioritize exact matches from within specific columns like 'Programmer Notes', 'Description', etc. Return only the most relevant snippet, not the entire row.

Documentation:
{context_str}

User question:
{user_query}

Answer based only on the documentation above."""

    answer = call_ollama(prompt_template)

    print(f"DEBUG -- Retrieved Context:\n{context_str}\n")
    print(f"DEBUG -- Full Prompt Sent to LLM:\n{prompt_template}\n")
    print(f"DEBUG -- Model Output:\n{answer}")

    return answer.strip()

if __name__ == "__main__":
    excel_dir = "excel_files/"  # Directory containing the Excel files
    print("Loading Excel files...")
    docs, metadata_list = load_documents(excel_dir)
    print(f"{len(docs)} document chunks loaded.")

    print("Generating embeddings...")
    embed_model, doc_embeddings_np_array = embed_docs(docs)

    print("Building FAISS vector index...")
    faiss_index_obj = build_faiss_index(doc_embeddings_np_array)

    user_question_input = input("\nAsk your Redshift schema question:\n")

    answer_response_output = answer_query(
        user_question_input,
        embed_model,
        faiss_index_obj,
        docs
    )

    print("\n=== ANSWER ===\n")
    print(answer_response_output)

