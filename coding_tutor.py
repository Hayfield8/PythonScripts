import os
import json
import pickle
import subprocess
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# ------------------------------
# CONFIG
# ------------------------------
DATA_DIR = r"C:\Users\m_hay\CodingTutor\data"
EMBED_DIR = r"C:\Users\m_hay\CodingTutor\embeddings"
LOG_DIR = r"C:\Users\m_hay\CodingTutor\logs"

MODEL_NAME = "llama3:8b"
SYSTEM_PROMPT = "You are an expert coding tutor. Answer step by step, give examples, and explain code clearly."

TOP_K = 3         # Number of context documents to retrieve
CONTEXT_TOKENS = 2048  # Ollama context size

# ------------------------------
# SETUP
# ------------------------------
os.makedirs(EMBED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load or create FAISS index
index_file = os.path.join(EMBED_DIR, "index.faiss")
meta_file = os.path.join(EMBED_DIR, "metadata.pkl")

if os.path.exists(index_file):
    index = faiss.read_index(index_file)
    with open(meta_file, "rb") as f:
        metadata = pickle.load(f)
else:
    index = faiss.IndexFlatL2(384)  # all-MiniLM-L6-v2 outputs 384-dim vectors
    metadata = []

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def extract_code_comments(text, ext):
    if ext == '.py':
        # Extract docstrings and comments
        comments = re.findall(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'|#(.*)', text, re.DOTALL)
        # Flatten matches
        comments = [c for grp in comments for c in grp if c]
        return "\n".join(comments)
    elif ext in ('.js', '.java'):
        # Extract // and /* */ comments
        comments = re.findall(r'//(.*)|/\*(.*?)\*/', text, re.DOTALL)
        comments = [c for grp in comments for c in grp if c]
        return "\n".join(comments)
    else:
        return text

def load_documents():
    docs = []
    filenames = []
    for fname in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, fname)
        if fname.endswith(('.txt', '.md')):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                docs.append(content)
                filenames.append(fname)
        elif fname.endswith('.pdf'):
            try:
                text = ""
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"
                docs.append(text)
                filenames.append(fname)
            except Exception as e:
                print(f"Failed to read {fname}: {e}")
        elif fname.endswith(('.py', '.js', '.java')):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                comments = extract_code_comments(content, os.path.splitext(fname)[1])
                docs.append(content + "\n\n" + comments)
                filenames.append(fname)
        elif ext == '.docx':
            try:
                doc = Document(path)
                full_text = "\n".join([p.text for p in doc.paragraphs])
                docs.append(full_text)
                filenames.append(fname)
            except Exception as e:
                print(f"Failed to read {fname}: {e}")
    return docs, filenames

def embed_and_index(new_texts, new_meta):
    vectors = embed_model.encode(new_texts, show_progress_bar=True)
    index.add(vectors)
    metadata.extend(new_meta)
    faiss.write_index(index, index_file)
    with open(meta_file, "wb") as f:
        pickle.dump(metadata, f)

def retrieve_context(query, top_k=TOP_K):
    q_vec = embed_model.encode([query])
    if len(metadata) == 0:
        return []
    D, I = index.search(q_vec, top_k)
    results = []
    for i in I[0]:
        if i < len(metadata):
            fname = metadata[i]
            fpath = os.path.join(DATA_DIR, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                results.append(f.read())
    return results

def log_interaction(user_prompt, model_response):
    log_data = {
        "user": user_prompt,
        "model": model_response
    }
    log_file = os.path.join(LOG_DIR, f"{len(os.listdir(LOG_DIR))+1}.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    # Also add this interaction to embeddings
    embed_and_index([user_prompt + " " + model_response], [log_file])

def ask_ollama(query, context_docs):
    context_text = "\n\n".join(context_docs)

    if context_text.strip():
        full_system = (
            SYSTEM_PROMPT
            + "\n\nYou also have access to some additional user-provided documents. "
            + "Use them only if they are relevant:\n"
            + context_text
        )
    else:
        full_system = SYSTEM_PROMPT

    # Build one prompt string
    full_prompt = full_system + "\n\nUser question:\n" + query

    # Call Ollama with just the model + prompt
    cmd = ["ollama", "run", MODEL_NAME, full_prompt]

    # Run and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    response = result.stdout.strip()
    return response if response else "[No response received]"

# ------------------------------
# INITIALIZE DOCUMENTS
# ------------------------------
# Only index documents once
if len(metadata) == 0:
    docs, fnames = load_documents()
    if docs:
        embed_and_index(docs, fnames)

# ------------------------------
# MAIN LOOP
# ------------------------------
print("Local Coding Tutor ready! Type 'exit' to quit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break
    context = retrieve_context(user_input)
    answer = ask_ollama(user_input, context)
    print(f"Tutor:\n{answer}\n")
    log_interaction(user_input, answer)
