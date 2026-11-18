import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Groq (rapide, API cloud)
from groq import Groq

load_dotenv()  # Charge les variables depuis le fichier .env

# ------------------------
# Configuration
# ------------------------
PDF_PATH = os.environ.get("KNOWLEDGE_PDF", "knowledge.pdf")
MODEL_NAME = "qwen/qwen3.8-27b"
TOP_K = 2                 # nombre de passages retrouvés
MAX_CONTEXT_CHARS = 2000  # assez pour que le modèle voie les 2 passages (chunk_size = 800)
STOPPERS = ["Question", "Note :", "Remarque :", "Article", "En résumé"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", "METS_TA_CLE_ICI"))


class ChatRequest(BaseModel):
    message: str


# ------------------------
# Charger PDF + vector store
# ------------------------
print("Chargement du PDF...")

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

print(f"PDF chargé et indexé ✅ ({len(documents)} pages, {len(docs)} passages)")


# ------------------------
# Recherche intelligente
# ------------------------
def clean_chunk(text):
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


def search_docs(query, k=TOP_K):
    """Retourne les k passages les plus proches, avec leur score de pertinence (0 à 1)."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    passages = []
    for doc, distance in results:
        # FAISS renvoie une distance L2 au carré ; les vecteurs de all-MiniLM-L6-v2
        # étant normalisés, similarité cosinus = 1 - distance / 2
        score = 1 - float(distance) / 2
        page = doc.metadata.get("page")
        passages.append({
            "text": doc.page_content.strip(),           # texte original, affiché dans l'interface
            "context": clean_chunk(doc.page_content),   # texte nettoyé, envoyé au modèle
            "source": os.path.basename(doc.metadata.get("source", PDF_PATH)),
            "page": page + 1 if isinstance(page, int) else None,
            "score": round(max(0.0, min(1.0, float(score))), 3),
        })
    return passages


# ------------------------
# Prompt
# ------------------------
def build_prompt(context, question):
    context = context[:MAX_CONTEXT_CHARS]
    return f"""Tu es un assistant support IT.
RÈGLES STRICTES :
- Utilise UNIQUEMENT les informations du contexte ci-dessous.
- Recopie les étapes telles quelles, sans reformuler ni interpréter.
- Un titre comme "Gestionnaire Des Tâches" est une ACTION à faire : formule-le comme "Ouvre le Gestionnaire des tâches".
- Ne change JAMAIS le sens d'un terme technique.
- Si la réponse n'est pas dans le contexte, réponds : "Je ne sais pas."

Contexte :
{context}

Question :
{question}

Réponse (étapes numérotées, fidèles au contexte) :"""


def event(payload):
    """Une ligne JSON par événement (NDJSON), lue en streaming par le frontend."""
    return json.dumps(payload, ensure_ascii=False) + "\n"


# ------------------------
# Pages
# ------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/status")
def status():
    return {
        "document": os.path.basename(PDF_PATH),
        "pages": len(documents),
        "passages": len(docs),
        "model": MODEL_NAME,
    }


# ------------------------
# Chat (streaming NDJSON)
#   {"type": "sources", "sources": [...]}
#   {"type": "token", "text": "..."}   (plusieurs fois)
#   {"type": "done"}  ou  {"type": "error", "message": "..."}
# ------------------------
@app.post("/chat")
def chat(request: ChatRequest):

    def generate():
        try:
            msg = request.message.lower()
            print("📩 Question:", request.message)

            # Règles simples
            if "mediametrie" in msg and ("mdp" in msg or "mot de passe" in msg):
                yield event({"type": "sources", "sources": []})
                yield event({"type": "token", "text": "1. Ouvre Jira\n2. Clique sur créer\n3. Suis la procédure interne\n"})
                yield event({"type": "done"})
                return

            # RAG : recherche dans le PDF
            passages = search_docs(request.message)
            yield event({"type": "sources", "sources": passages})

            context = "\n\n".join(p["context"] for p in passages)
            prompt = build_prompt(context, request.message)

            stream = groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un assistant support IT helpdesk. Tu réponds uniquement en français, avec des étapes numérotées claires.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=300,
                stop=["Question", "Note :", "Remarque :", "En résumé"],  # max 4 items (limite Groq)
                stream=True,
            )

            # On garde une petite réserve de caractères pour ne jamais
            # envoyer le début d'un mot parasite (ex. "Remarque :").
            hold = max(len(s) for s in STOPPERS)
            buffer, sent = "", 0
            for chunk in stream:
                buffer += chunk.choices[0].delta.content or ""
                cuts = [buffer.find(s) for s in STOPPERS if s in buffer]
                if cuts:
                    buffer = buffer[:min(cuts)]
                    break
                safe = len(buffer) - hold
                if safe > sent:
                    yield event({"type": "token", "text": buffer[sent:safe]})
                    sent = safe

            buffer = buffer.rstrip()
            if len(buffer) > sent:
                yield event({"type": "token", "text": buffer[sent:]})

            if not buffer.strip():
                yield event({"type": "token", "text": "❌ Aucune réponse du modèle"})

            print("✅ Réponse:", buffer[:100])
            yield event({"type": "done"})

        except Exception as e:
            print("❌ ERREUR:", e)
            yield event({"type": "error", "message": f"Erreur serveur : {e}"})

    return StreamingResponse(generate(), media_type="application/x-ndjson")
