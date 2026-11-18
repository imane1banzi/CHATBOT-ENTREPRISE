FROM python:3.12-slim

# Hugging Face Spaces exécute le conteneur avec l'utilisateur 1000
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
RUN mkdir -p uploads && chown user uploads

USER user
ENV HOME=/home/user

# Télécharge le modèle d'embeddings pendant le build (démarrage plus rapide)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
