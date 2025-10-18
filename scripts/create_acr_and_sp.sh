#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 --acr-name NAME [--resource-group RG] [--location LOCATION] [--sku {Basic,Standard,Premium}] [--no-sp]

Creates an Azure Container Registry (ACR) and optionally a service principal with acrpush permissions
so CI (GitHub Actions) can push images.

Options:
  --acr-name NAME        Short ACR name (e.g. myregistry) -> login server will be myregistry.azurecr.io
  --resource-group RG    Resource group to use/create (default: rag-ai-foundations-demo-rg)
  --location LOCATION    Azure region (default: eastus)
  --sku SKU              ACR SKU: Basic, Standard or Premium (default: Standard)
  --no-sp                Do not create a service principal (default: create SP)
  -h, --help             Show this help

Outputs printed on success (copy these to GitHub secrets):
  ACR_LOGIN_SERVER, ACR_NAME, ACR_SP_APPID, ACR_SP_PASSWORD, AZURE_TENANT_ID, AZURE_SUBSCRIPTION_ID, RESOURCE_GROUP

EOF
}

ACR_NAME=""
RG="rag-ai-foundations-demo-rg"
LOCATION="eastus"
SKU="Standard"
CREATE_SP=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --acr-name) ACR_NAME="$2"; shift 2;;
    --resource-group) RG="$2"; shift 2;;
    --location) LOCATION="$2"; shift 2;;
    --sku) SKU="$2"; shift 2;;
    --no-sp) CREATE_SP=0; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

if [[ -z "$ACR_NAME" ]]; then
  echo "--acr-name is required" >&2
  usage
  exit 2
fi

echo "Checking Azure CLI login..."
if ! az account show > /dev/null 2>&1; then
  echo "Not logged in. Run: az login" >&2
  exit 3
fi

SUBSCRIPTION_ID=$(az account show --query id -o tsv)
TENANT_ID=$(az account show --query tenantId -o tsv)

echo "Using subscription: $SUBSCRIPTION_ID (tenant: $TENANT_ID)"

echo "Creating or ensuring resource group '$RG' in $LOCATION..."
az group create --name "$RG" --location "$LOCATION" --output none

echo "Creating ACR '$ACR_NAME' (sku=$SKU) in resource group '$RG'..."
az acr create --name "$ACR_NAME" --resource-group "$RG" --sku "$SKU" --admin-enabled false --output none

LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --resource-group "$RG" --query loginServer -o tsv)
echo "ACR login server: $LOGIN_SERVER"

if [[ $CREATE_SP -eq 1 ]]; then
  echo "Creating a service principal for ACR push access..."
  # Create an SP with limited scope for the registry
  RG_SCOPE="/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RG/providers/Microsoft.ContainerRegistry/registries/$ACR_NAME"

  # Use a stable name for the SP to make reruns idempotent
  SP_NAME="sp-${ACR_NAME}-ci"

  # Try to remove existing sp credentials if present (do not delete SP itself automatically)
  EXISTING=$(az ad sp list --display-name "$SP_NAME" --query "length([])" -o tsv)
  if [[ "$EXISTING" -gt 0 ]]; then
    echo "Service principal '$SP_NAME' already exists. Creating a new credential instead."
    # create a password credential (client secret) for existing SP
    APP_ID=$(az ad sp show --id http://$SP_NAME --query appId -o tsv || true)
    if [[ -z "$APP_ID" ]]; then
      # fallback: find SP by displayName
      APP_ID=$(az ad sp list --display-name "$SP_NAME" --query "[0].appId" -o tsv)
    fi
    if [[ -z "$APP_ID" ]]; then
      echo "Could not locate existing SP. Will create a new one." >&2
      EXISTING=0
    else
      echo "Found existing SP appId=$APP_ID"
    fi
  fi

  if [[ "$EXISTING" -eq 0 ]]; then
    # Create new SP and assign acrpush role scoped to the registry
    SP_JSON=$(az ad sp create-for-rbac --name "http://$SP_NAME" --scopes "$RG_SCOPE" --role acrpush -o json)
    APP_ID=$(echo "$SP_JSON" | python -c "import sys, json; print(json.load(sys.stdin)['appId'])")
    PASSWORD=$(echo "$SP_JSON" | python -c "import sys, json; print(json.load(sys.stdin)['password'])")
    echo "Created SP with appId=$APP_ID"
  else
    # Create a new credential for the existing app
    CRED_JSON=$(az ad app credential reset --id "$APP_ID" --append --query "{appId:appId,password:password}" -o json)
    PASSWORD=$(echo "$CRED_JSON" | python -c "import sys, json; print(json.load(sys.stdin)['password'])")
    echo "Created new credential for existing appId=$APP_ID"
  fi

  # Ensure role assignment exists (sometimes create-for-rbac already assigned)
  echo "Ensuring role assignment 'acrpull/acrpush' exists for SP on the registry scope..."
  az role assignment create --assignee "$APP_ID" --role acrpush --scope "$RG_SCOPE" --output none || true

  echo
  echo "Service principal created. Values to add to GitHub secrets:" 
  echo "  ACR_NAME=$ACR_NAME"
  echo "  ACR_LOGIN_SERVER=$LOGIN_SERVER"
  echo "  ACR_SP_APPID=$APP_ID"
  echo "  ACR_SP_PASSWORD=$PASSWORD"
  echo "  AZURE_TENANT_ID=$TENANT_ID"
  echo "  AZURE_SUBSCRIPTION_ID=$SUBSCRIPTION_ID"
  echo "  RESOURCE_GROUP=$RG"
else
  echo
  echo "ACR created. Values to add to GitHub secrets:" 
  echo "  ACR_NAME=$ACR_NAME"
  echo "  ACR_LOGIN_SERVER=$LOGIN_SERVER"
  echo "  AZURE_TENANT_ID=$TENANT_ID"
  echo "  AZURE_SUBSCRIPTION_ID=$SUBSCRIPTION_ID"
  echo "  RESOURCE_GROUP=$RG"
fi

echo
echo "Done. You can now run scripts/push_to_acr.py or update your GitHub workflow secrets with the values above."
