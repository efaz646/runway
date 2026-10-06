# Runway as a container (Render, Railway, Fly.io, Google Cloud Run, or your own server).
# Pass the keys at run time; they are never baked into the image:
#   docker build -t runway .
#   docker run -p 8501:8501 -e GEMINI_API_KEY=... -e ELEVENLABS_API_KEY=... runway
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PORT=8501
EXPOSE 8501
HEALTHCHECK CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/_stcore/health')"
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
