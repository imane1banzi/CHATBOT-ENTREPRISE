from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# ✅ Groq (rapide, gratuit, API cloud)
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()  # ✅ Charge les variables depuis le fichier .env

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
# Groq client
# ✅ Mets ta clé API ici ou dans une variable d'environnement GROQ_API_KEY
# ------------------------
groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY", "METS_TA_CLE_ICI")
)

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
    chunk_size=800,
    chunk_overlap=100
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
    return templates.TemplateResponse(request=request, name="index.html")

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
            # RAG : recherche dans le PDF
            # ------------------------
            context = search_docs(request.message)
            print("📄 Context (200 chars):", context[:200])

            prompt = build_prompt(context, request.message)

            # ✅ Appel Groq — ultra rapide (Llama 3 sur GPU dédié)
            completion = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",  # ✅ successeur de llama3-8b-8192
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un assistant support IT helpdesk. Tu réponds uniquement en français, avec des étapes numérotées claires."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=300,
                stop=["Question", "Note :", "Remarque :", "En résumé"]  # ✅ max 4 items (limite Groq)
            )

            response = completion.choices[0].message.content.strip()

            # Sécurité : coupe tout texte parasite après la réponse
            for stopper in ["Question", "Note :", "Remarque :", "Article", "En résumé"]:
                if stopper in response:
                    response = response[:response.index(stopper)].strip()

            if not response:
                yield "❌ Aucune réponse du modèle"
            else:
                print("✅ Réponse:", response[:100])
                yield response

        except Exception as e:
            print("❌ ERREUR:", e)
            yield f"❌ Erreur serveur: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")