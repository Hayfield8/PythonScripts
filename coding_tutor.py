import os
import json
import pickle
import subprocess
import pdfplumber
import re
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

    print(f"[DEBUG] Scanning data directory: {DATA_DIR}")
    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            path = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            print(f"[DEBUG] Found file: {path}")

            try:
                if ext in ('.txt', '.md'):
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    docs.append(content)
                    filenames.append(path)
                    print(f"[DEBUG] Loaded text/markdown: {path} ({len(content)} chars)")

                elif ext == '.pdf':
                    text = ""
                    with pdfplumber.open(path) as pdf:
                        for page in pdf.pages:
                            text += (page.extract_text() or "") + "\n"
                    docs.append(text)
                    filenames.append(path)
                    print(f"[DEBUG] Loaded PDF: {path} ({len(text)} chars)")

                elif ext in ('.py', '.js', '.java'):
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    comments = extract_code_comments(content, ext)
                    docs.append(content + "\n\n" + comments)
                    filenames.append(path)
                    print(f"[DEBUG] Loaded code file: {path} ({len(content)} chars + comments)")

                elif ext == '.docx':
                    from docx import Document
                    doc = Document(path)
                    full_text = "\n".join([p.text for p in doc.paragraphs])
                    docs.append(full_text)
                    filenames.append(path)
                    print(f"[DEBUG] Loaded DOCX: {path} ({len(full_text)} chars)")

                else:
                    print(f"[DEBUG] Skipped unsupported file type: {path}")

            except Exception as e:
                print(f"[ERROR] Failed to process {path}: {e}")

    print(f"[DEBUG] Total loaded documents: {len(docs)}")
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
            + "\n\n[Extra documents provided by the user — use only if relevant:]\n"
            + context_text
        )
    else:
        full_system = SYSTEM_PROMPT

    # Build the full prompt
    full_prompt = (
        full_system
        + "\n\n[User Question:]\n"
        + query
    )

    # 🔍 Show what we're sending to Ollama
    print("\n" + "="*40)
    print("DEBUG: Sending prompt to Ollama:")
    print("="*40)
    print(full_prompt)
    print("="*40 + "\n")

    # Run Ollama
    cmd = ["ollama", "run", MODEL_NAME, full_prompt]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # 🔍 Show raw stdout / stderr
    print("DEBUG: Raw Ollama stdout:")
    print(result.stdout if result.stdout else "[empty]")
    print("DEBUG: Raw Ollama stderr:")
    print(result.stderr if result.stderr else "[empty]")

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
