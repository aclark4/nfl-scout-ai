# NFL Scout AI - FastAPI + Chroma + Claude
FROM python:3.11-slim

WORKDIR /app

# Runtime dependencies only (nfl_data_py is only needed to rebuild the stats JSON)
RUN pip install --no-cache-dir anthropic chromadb fastapi uvicorn python-dotenv pandas

COPY qb_stats.py documents.py retriever.py tools.py agent.py app.py ./
COPY data/qb_stats_2024.json data/qb_stats_2024.json

# Build the vector index at image build time (also caches the embedding model)
RUN python retriever.py --rebuild

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
