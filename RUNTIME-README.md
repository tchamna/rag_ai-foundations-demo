Runtime image (minimal)

This repository contains a minimal runtime Docker image that runs the Streamlit UI and uses the precomputed FAISS vectorstore in `vectorstore/`.

Purpose
- Provide a smaller image (~800–900 MB) that runs the UI and uses precomputed embeddings (no heavy ML model installs).

How to build locally (PowerShell)

```powershell
# Build the image locally
docker build -f Dockerfile.runtime -t rag_ai_backend:runtime-minimal .

# Run it locally and map port 8000
docker run -d --name rag_runtime -p 8000:8000 rag_ai_backend:runtime-minimal

# View logs
docker logs -f rag_runtime
```

How to push to Azure Container Registry (ACR)

Option A (local push - needs az login and ACR login):

```powershell
# Tag the image for your ACR
docker tag rag_ai_backend:runtime-minimal <ACR_LOGIN_SERVER>/rag_ai_backend:runtime-minimal

# Login to ACR (after az login)
az acr login --name <ACR_NAME>

# Push
docker push <ACR_LOGIN_SERVER>/rag_ai_backend:runtime-minimal
```

Option B (recommended) — build in ACR (avoids large upload):

```powershell
az acr build --registry <ACR_NAME> --image rag_ai_backend:runtime-minimal -f Dockerfile.runtime .
```

GitHub Actions

A sample GitHub Actions workflow is included at `.github/workflows/build_and_push_runtime_acr.yml` that uses `az acr build` from a runner. Provide these secrets in your repo (or use OIDC federated credential):

- `AZURE_CLIENT_ID` — app registration client id (if not using OIDC)
- `AZURE_TENANT_ID` — tenant id
- `AZURE_CLIENT_SECRET` — client secret (if not using OIDC)
- `ACR_NAME` — ACR resource name (e.g. `ragfoundationsdemoregistry`)
- `ACR_LOGIN_SERVER` — login server (e.g. `ragfoundationsdemoregistry.azurecr.io`)
- `AZURE_APP_NAME` and `AZURE_RESOURCE_GROUP` — optional, to automatically update App Service container settings

Troubleshooting
- If `docker push` fails due to network resets, use `az acr build` to build in Azure.
- If OIDC login fails in Actions, ensure the federated identity credential exists in the Azure App registration and the subject matches `repo:OWNER/REPO:ref:refs/heads/main` (or appropriate branch).