DEPLOY_AZURE
============

This document explains the recommended way to deploy the app to Azure App Service and common troubleshooting steps (Kudu). It includes the exact startup command to paste into the Azure Portal Startup Command for a Streamlit app.

Recommended approach (no heavy ML packages)
-------------------------------------------
1) Precompute the FAISS vectorstore locally and commit the `vectorstore/` folder to your repository. This avoids installing `sentence-transformers` / `torch` on App Service.

2) Push your repository to GitHub and configure Azure App Service to pull from the repo (GitHub Actions or direct Git deployment).

3) Configure the App Service startup command (use the full line below in the Portal -> Configuration -> Startup Command):

```text
python -m streamlit run app.py --server.port %PORT% --server.address 0.0.0.0
```

Note: Azure injects the port number via the `PORT` environment variable. Use `--server.address 0.0.0.0` to bind externally.

Kudu / runtime diagnostics
--------------------------
If the app crashes in Azure, use Kudu (Advanced Tools) to inspect the runtime and logs.

Helpful Kudu/SSH commands (run from Kudu debug console or SSH into the App Service container):

- Show Python version and executable path

```powershell
python -V
where python
```

- Print where the `xml` stdlib module resolves (useful if `openpyxl` raises ModuleNotFoundError: No module named 'xml')

```powershell
python -c "import xml; print(getattr(xml,'__file__',None))"
```

- List installed pip packages (look for a stray `xml` package that shadows the stdlib):

```powershell
python -m pip list --format=columns
```

- Tail App Service logs (from your machine if az CLI is configured):

```powershell
az webapp log tail --resource-group <RG_NAME> --name <APP_NAME>
```

- Check the runtime temp logs (common locations):
  - `D:\local\Temp\rag_runtime_errors.log`
  - `D:\home\LogFiles\Application\` and files inside

If you find a top-level `xml` package in `pip list`, uninstall it by editing your deployment or removing it from `requirements.txt`. The stdlib `xml` package must not be shadowed by a pip package named `xml`.

If you'd rather include heavy ML packages instead of precomputing the vectorstore (not recommended on low-memory App Service plans), add these to `requirements.txt` before deployment:

- sentence-transformers
- torch
- transformers (optional for local generators)

Also note: installing `torch` on App Service can take a long time and may exceed build time limits.

