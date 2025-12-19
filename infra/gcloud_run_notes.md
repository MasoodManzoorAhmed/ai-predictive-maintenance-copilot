# Google Cloud Run Deployment Notes (CMAPSS Predictive Maintenance Copilot)

This document captures the deployment setup and commands used to build and deploy:
- **cmaps-api** (FastAPI backend)
- **cmaps-ui** (Streamlit frontend)

Target platform: **Google Cloud Run**  
Region used: **europe-west2**  
Container Registry: **Artifact Registry (Docker images)**

---

## 1) Services Overview

### Backend (API) — `cmaps-api`
- Serves prediction endpoints (FD001–FD004) and Copilot endpoints.
- Runs as a container on Cloud Run.
- Exposed publicly (allow unauthenticated) for demo purposes.

### Frontend (UI) — `cmaps-ui`
- Streamlit dashboard and Copilot UI.
- Runs as a container on Cloud Run.
- Calls the backend using environment variable:
  - `API_BASE_URL=<Cloud Run URL of cmaps-api>`

---

## 2) Key Environment Variables

### UI (Streamlit)
- `API_BASE_URL` → Cloud Run URL of the API service (e.g. https://cmaps-api-xxxx.a.run.app)

### API (FastAPI)
Typical environment variables may include:
- Model/config paths (if applicable)
- Feature flags (e.g. RAG enabled)
- LLM keys (must be stored as env vars, never hardcoded)

> Note: Keep secrets in env vars (Cloud Run service env vars) or Secret Manager.
> Do not commit API keys to GitHub.

---

## 3) Build & Deploy Flow

We use:
- **Cloud Build** to build Docker images using YAML configs in `infra/`
- **Cloud Run** to deploy those images as services

### Streamlit UI (worked command example)
Build:
- `gcloud builds submit --config infra/cloudbuild.streamlit.yaml .`

Deploy:
- `gcloud run deploy cmaps-ui --image <IMAGE_URI> --region europe-west2 --allow-unauthenticated --port 8080 --set-env-vars API_BASE_URL=<API_URL>`

---

## 4) Verification Checklist

After deployment:
1. Confirm API is reachable:
   - Open API URL in browser OR call a health endpoint (if implemented)
2. Confirm UI loads:
   - Open UI URL
3. Confirm UI → API connectivity:
   - UI predictions should succeed
   - Copilot endpoint calls should return response

---

## 5) Notes / Common Issues

- If the UI shows “connection not private” on mobile:
  - Confirm you are using the Cloud Run HTTPS URL (not a local IP).
  - This can also happen due to device time mismatch or DNS caching.
- If UI cannot call API:
  - Verify `API_BASE_URL` is set correctly on **cmaps-ui**
  - Verify API is deployed and publicly accessible (or configure IAM if private).

---

## 6) Change Management

Before building and deploying:
- Remove unused files / comments (done)
- Run locally to confirm nothing broke
- Commit clean changes to Git
- Rebuild and redeploy UI/API containers

---

## 7) Current Cloud Run Endpoints (Reference)

> ⚠️ These URLs are **deployment references**.
> They may change if the services are redeployed, renamed, or moved to another project/region.

### Backend API (FastAPI)
- Service name: `cmaps-api`
- Cloud Run URL:
  https://cmaps-api-419391401356.europe-west2.run.app

### Frontend UI (Streamlit)
- Service name: `cmaps-ui`
- Cloud Run URL:
  https://cmaps-ui-419391401356.europe-west2.run.app

### UI → API Connectivity
- The UI calls the backend using:
  ```bash
  API_BASE_URL=https://cmaps-api-419391401356.europe-west2.run.app
