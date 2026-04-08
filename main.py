from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ✅ Fix warning : utiliser langchain-huggingface
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

import requests

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
templates = Jinja2Templates(directory="templates")

# ------------------------
# Schema
# ------------------------
class ChatRequest(BaseModel):
    message: str

# ------------------------
# Charger PDF + vector store
# ------------------------
print("Chargement du PDF...")

loader = PyPDFLoader("knowledge.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,    # ✅ augmenté pour ne pas couper les procédures
    chunk_overlap=100  # ✅ augmenté pour garder le contexte entre chunks
)

docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(docs, embeddings)

print("PDF chargé et indexé ✅")

# ------------------------
# Recherche intelligente
# ------------------------
def search_docs(query):
    results = vectorstore.similarity_search(query, k=2)
    chunks = []
    for doc in results:
        lines = doc.page_content.split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # ✅ Titre en majuscules → reformulé comme action claire
            if line.isupper():
                cleaned_lines.append(f"Étape : {line.title()}")
            else:
                cleaned_lines.append(line)
        chunks.append("\n".join(cleaned_lines))
    text = "\n\n".join(chunks)
    return text.replace("  ", " ")

# ------------------------
# Prompt
# ------------------------
def build_prompt(context, question):
    # ✅ Contexte tronqué à 800 caractères max
    context = context[:800]

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

# ------------------------
# Page UI
# ------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ------------------------
# Chat
# ------------------------
@app.post("/chat")
def chat(request: ChatRequest):

    def generate():
        try:
            msg = request.message.lower()
            print("📩 Question:", request.message)

            # ------------------------
            # Règles simples
            # ------------------------
            if "mediametrie" in msg and ("mdp" in msg or "mot de passe" in msg):
                yield "1. Ouvre Jira\n2. Clique sur créer\n3. Suis la procédure interne\n"
                return

            # ------------------------
            # RAG
            # ------------------------
            context = search_docs(request.message)
            print("📄 Context (200 chars):", context[:200])

            prompt = build_prompt(context, request.message)

            payload = {
                "model": "phi3:latest",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 150,
                    "num_ctx": 1024,
                },
                # ✅ Stop tokens pour éviter que le modèle continue après la réponse
                "stop": ["Question", "Note", "Remarque", "Article", "En résumé", "---"]
            }

            # ✅ Timeout augmenté à 300 secondes
            r = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=300
            )

            if r.status_code != 200:
                yield f"❌ Ollama error: {r.text}"
                return

            data = r.json()
            response = data.get("response", "").strip()

            # ✅ Sécurité : coupe tout texte parasite après la réponse
            for stopper in ["Question", "Note :", "Remarque :", "Article", "En résumé"]:
                if stopper in response:
                    response = response[:response.index(stopper)].strip()

            if not response:
                yield "❌ Aucune réponse du modèle"
            else:
                print("✅ Réponse:", response[:100])
                yield response

        except requests.exceptions.Timeout:
            print("❌ TIMEOUT Ollama")
            yield "❌ Le modèle met trop de temps à répondre. Réessaie dans quelques secondes."

        except Exception as e:
            print("❌ ERREUR:", e)
            yield f"❌ Erreur serveur: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")