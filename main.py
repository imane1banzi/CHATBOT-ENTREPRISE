from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory="templates")

# Schema
class ChatRequest(BaseModel):
    message: str

# ------------------------
# Réponses métier (règles)
# ------------------------
def get_response(message: str):
    msg = message.lower()

    # 🔥 Cas Mediametrie mot de passe
    if "mediametrie" in msg and ("mot de passe" in msg or "mdp" in msg):
        return """1. Ouvre Jira et connecte-toi avec ton compte.
2. Clique sur "Créer" pour générer un nouveau ticket.
3. Dans le type de ticket, choisis "Demande".
4. Sélectionne le workspace Mediametrie.
5. Renseigne le résumé : "Mot de passe expiré – accès Mediametrie".
6. Remplis la description avec les détails du problème.
7. Affecte le ticket à toi-même.
8. Clique sur "Créer / Submit".
9. Note le numéro du ticket pour le suivi."""
    if "voxco" in msg and ("mot de passe" in msg or "mdp" in msg):
        return """1. Ouvre Jira et connecte-toi avec ton compte.
2. Clique sur "Créer" pour ouvrir un nouveau ticket.
3. Dans le type de ticket, sélectionne "Demande".
4. Choisis le service ou workspace "Voxco".
5. Sélectionne l’option "Changement de mot de passe".
6. Ajoute un commentaire en décrivant brièvement le problème rencontré.
7. Affecte le ticket à toi-même.
8. Vérifie les informations saisies puis clique sur "Créer / Submit".
9. Note le numéro du ticket pour assurer le suivi."""
    # ❌ inconnu
    return "Je ne sais pas encore répondre à cette question."
    # 🔥 Cas Voxco mot de passe
 
# ------------------------
# Page UI
# ------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ------------------------
# Chat API
# ------------------------
@app.post("/chat")
def chat(request: ChatRequest):
    response = get_response(request.message)
    return {"response": response}