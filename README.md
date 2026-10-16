# OSCC GraphRAG deployment package

This package converts the notebook into a persistent FastAPI web app with its existing embedded frontend.

## Files you must add

Place these existing artifacts in `models/`:

- `oral_cancer_efficientnetb0.keras` — trained EfficientNetB0 model
- `oscc_literature.faiss` — FAISS literature index
- `chunk_metadata.json` — metadata matching the FAISS index

Do **not** upload API keys or database passwords to GitHub.

## Environment variables

Configure these in your hosting provider's secret/environment settings:

- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE` (optional; defaults to `neo4j`)
- `GROQ_API_KEY`
- `GROQ_MODEL` (optional; defaults to `openai/gpt-oss-20b`)

Optional asset path overrides: `MODEL_PATH`, `FAISS_PATH`, `METADATA_PATH`.

## Run locally

```bash
docker build -t oscc-graphrag .
docker run --rm -p 7860:7860 \
  -e NEO4J_URI="..." -e NEO4J_USERNAME="..." -e NEO4J_PASSWORD="..." \
  -e NEO4J_DATABASE="neo4j" -e GROQ_API_KEY="..." \
  oscc-graphrag
```

Open `http://localhost:7860/`. API docs are at `/docs`; health check is `/health`.

## Deploy as a link

This is a Docker-based app, so it can be deployed on a persistent container host or a Hugging Face Space configured with the Docker SDK. Upload the package and model artifacts, configure the environment variables/secrets, and the host will build and run the app. The public URL will remain available according to the host's uptime/sleep policy.

## Important

- The original notebook used Google Drive paths, Colab Secrets, and an ngrok tunnel. Those are runtime-specific and are no longer used here.
- The app requires your existing trained model, FAISS index, metadata JSON, a reachable Neo4j database, and a valid Groq API key.
- This is a research/academic tool, not a clinical diagnostic system. Do not use it to make medical decisions.
