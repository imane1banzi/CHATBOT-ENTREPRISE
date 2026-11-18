import os
import re
import threading
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# LangChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Groq (API cloud, inférence rapide)
from groq import Groq

load_dotenv()  # Charge les variables depuis le fichier .env

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"   # documents fournis avec le projet
UPLOAD_DIR = BASE_DIR / "uploads"         # PDF importés depuis l'interface
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_MB = 20
TOP_K = 3
MAX_CONTEXT_CHARS = 3000
UNKNOWN_ANSWER = "Je ne sais pas."

app = FastAPI()

# ------------------------
# CORS
# ------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Templates
# ------------------------
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ------------------------
# Groq client (clé dans la variable d'environnement GROQ_API_KEY)
# ------------------------
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", "METS_TA_CLE_ICI"))


# ------------------------
# Schema
# ------------------------
class ChatRequest(BaseModel):
    message: str


# ------------------------
# Base de connaissances (FAISS)
# ------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = None
indexed_files: dict[str, int] = {}   # nom de fichier -> nombre de passages indexés
index_lock = threading.Lock()


def load_file(path: Path):
    """Charge un PDF / Markdown / texte et normalise la métadonnée 'source'."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        documents = PyPDFLoader(str(path)).load()
    elif suffix in (".md", ".txt"):
        documents = TextLoader(str(path), encoding="utf-8").load()
    else:
        return []

    for doc in documents:
        doc.metadata["source"] = path.name
        # PyPDFLoader numérote les pages à partir de 0
        if "page" in doc.metadata:
            doc.metadata["page"] = int(doc.metadata["page"]) + 1
    return documents


def add_to_index(path: Path) -> int:
    """Découpe un fichier et l'ajoute à l'index. Renvoie le nombre de passages."""
    global vectorstore

    chunks = splitter.split_documents(load_file(path))
    if not chunks:
        return 0

    with index_lock:
        if vectorstore is None:
            vectorstore = FAISS.from_documents(chunks, embeddings)
        else:
            vectorstore.add_documents(chunks)
        indexed_files[path.name] = len(chunks)
    return len(chunks)


def load_initial_documents():
    print("Indexation de la base de connaissances...")
    for folder in (KNOWLEDGE_DIR, UPLOAD_DIR):
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() in (".pdf", ".md", ".txt"):
                try:
                    n = add_to_index(path)
                    print(f"  - {path.name} : {n} passages")
                except Exception as e:
                    print(f"  ! {path.name} ignoré : {e}")
    print(f"Base prête : {len(indexed_files)} document(s) indexé(s)")


load_initial_documents()


# ------------------------
# Recherche
# ------------------------
def clean_chunk(text: str) -> str:
    cleaned_lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.isupper():
            cleaned_lines.append(f"Étape : {line.title()}")
        else:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).replace("  ", " ")


def format_source(metadata: dict) -> str:
    name = metadata.get("source", "document inconnu")
    page = metadata.get("page")
    return f"{name}, page {page}" if page else name


def search_docs(query: str):
    """Renvoie (contexte numéroté pour le prompt, liste des sources uniques)."""
    if vectorstore is None:
        return "", []

    results = vectorstore.similarity_search(query, k=TOP_K)

    blocks, sources = [], []
    for doc in results:
        label = format_source(doc.metadata)
        if label not in sources:
            sources.append(label)
        blocks.append(f"[Source : {label}]\n{clean_chunk(doc.page_content)}")

    return "\n\n".join(blocks)[:MAX_CONTEXT_CHARS], sources


# ------------------------
# Prompt
# ------------------------
def build_prompt(context, question):
    return f"""Tu es un assistant support IT.
RÈGLES STRICTES :
- Utilise UNIQUEMENT les informations du contexte ci-dessous.
- Recopie les étapes telles quelles, sans reformuler ni interpréter.
- Un titre comme "Gestionnaire Des Tâches" est une ACTION à faire : formule-le comme "Ouvre le Gestionnaire des tâches".
- Ne change JAMAIS le sens d'un terme technique.
- Si la réponse n'est pas dans le contexte, réponds exactement : "{UNKNOWN_ANSWER}"

Contexte :
{context}

Question :
{question}

Réponse (étapes numérotées, fidèles au contexte) :"""


# ------------------------
# Page UI
# ------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


# ------------------------
# Documents : liste + import de PDF
# ------------------------
@app.get("/documents")
def list_documents():
    return {"documents": [{"name": n, "chunks": c} for n, c in indexed_files.items()]}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés.")

    content = await file.read()
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Le fichier n'est pas un PDF valide.")
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"Fichier trop volumineux (max {MAX_UPLOAD_MB} Mo).")

    safe_name = re.sub(r"[^\w.\- ]", "_", filename)
    if safe_name in indexed_files:
        raise HTTPException(status_code=409, detail=f"« {safe_name} » est déjà indexé.")

    path = UPLOAD_DIR / safe_name
    path.write_bytes(content)

    try:
        chunks = add_to_index(path)
    except Exception as e:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Lecture du PDF impossible : {e}")

    if chunks == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Aucun texte extractible dans ce PDF (PDF scanné ?).",
        )

    print(f"📥 PDF importé : {safe_name} ({chunks} passages)")
    return {"name": safe_name, "chunks": chunks}


# ------------------------
# Chat
# ------------------------
@app.post("/chat")
def chat(request: ChatRequest):
    print("📩 Question:", request.message)

    try:
        context, sources = search_docs(request.message)
        if not context:
            return {
                "answer": "Aucun document n'est indexé. Importe un PDF pour commencer.",
                "sources": [],
            }

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un assistant support IT helpdesk. Tu réponds uniquement en français, avec des étapes numérotées claires.",
                },
                {"role": "user", "content": build_prompt(context, request.message)},
            ],
            temperature=0.1,
            max_tokens=400,
            stop=["Question", "Note :", "Remarque :", "En résumé"],  # max 4 (limite Groq)
        )

        answer = (completion.choices[0].message.content or "").strip()

        # Sécurité : coupe tout texte parasite après la réponse
        for stopper in ["Question", "Note :", "Remarque :", "Article", "En résumé", "[Source"]:
            if stopper in answer:
                answer = answer[: answer.index(stopper)].strip()

        if not answer:
            return {"answer": "❌ Aucune réponse du modèle", "sources": []}

        # Pas de source affichée quand le modèle ne sait pas
        if UNKNOWN_ANSWER.rstrip(".").lower() in answer.lower():
            sources = []

        print("✅ Réponse:", answer[:100], "| Sources:", sources)
        return {"answer": answer, "sources": sources}

    except Exception as e:
        print("❌ ERREUR:", e)
        return {"answer": f"❌ Erreur serveur: {e}", "sources": []}
