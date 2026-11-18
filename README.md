---
title: Chatbot Support IT
emoji: 🤖
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# 🤖 Chatbot Support IT (RAG)

Assistant de support IT qui répond **uniquement à partir de vos documents** et **cite ses sources** (fichier + page).

## Fonctionnalités

- 📄 **Import de PDF depuis l'interface** : les documents sont découpés et indexés à chaud, sans redémarrage
- 📎 **Sources citées** sous chaque réponse (`guide.pdf, page 3`)
- 🚫 Répond « Je ne sais pas » quand l'information n'est pas dans les documents
- 📚 Base de connaissances initiale chargée depuis `knowledge/` (`.pdf`, `.md`, `.txt`)

## Stack

| Rôle | Outil |
|---|---|
| API & interface | FastAPI, Jinja2, Tailwind |
| Découpage des documents | LangChain (`RecursiveCharacterTextSplitter`) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Recherche vectorielle | FAISS |
| LLM | Llama 3.1 8B via l'API Groq |

## Fonctionnement

1. Les documents sont découpés en passages de 800 caractères et vectorisés.
2. À chaque question, les 3 passages les plus proches sont retrouvés dans FAISS.
3. Le LLM répond en s'appuyant uniquement sur ces passages ; leurs sources sont renvoyées à l'interface.

## Lancer le projet

Il faut une clé API Groq gratuite : <https://console.groq.com/keys>.

```bash
pip install -r requirements.txt
echo "GROQ_API_KEY=ta_cle" > .env
uvicorn main:app --reload
```

Ou avec Docker :

```bash
docker build -t chatbot .
docker run -p 7860:7860 -e GROQ_API_KEY=ta_cle chatbot
```

## API

| Méthode | Route | Rôle |
|---|---|---|
| `POST` | `/chat` | `{"message": "..."}` → `{"answer": "...", "sources": [...]}` |
| `POST` | `/upload` | Import d'un PDF (multipart, champ `file`) |
| `GET` | `/documents` | Liste des documents indexés |
