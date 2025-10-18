from azure.identity import DefaultAzureCredential
from azure.mgmt.containerregistry import ContainerRegistryManagementClient
import os
import shlex
import subprocess

def az_subscription_id_from_cli() -> str:
    """Try to get the subscription id from az account show if not provided via env."""
    try:
        out = subprocess.check_output(shlex.split("az account show --query id -o tsv"), text=True)
        return out.strip()
    except Exception:
        return ""


def main():
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID") or az_subscription_id_from_cli()
    if not subscription_id:
        raise SystemExit("Subscription id not provided via AZURE_SUBSCRIPTION_ID and could not be detected from az CLI")

    resource_group = os.environ.get("AZURE_RESOURCE_GROUP", "rag-ai-foundations-demo-rg")
    registry_name = os.environ.get("ACR_NAME", "rag-ai-foundations-demo-registry")
    location = os.environ.get("AZURE_LOCATION", "eastus")

    # Authenticate (requires `az login` or managed identity)
    credential = DefaultAzureCredential()
    client = ContainerRegistryManagementClient(credential, subscription_id)

    print(f"Creating ACR '{registry_name}' in resource group '{resource_group}' (subscription {subscription_id})...")
    poller = client.registries.begin_create(
        resource_group_name=resource_group,
        registry_name=registry_name,
        registry={
            "location": location,
            "sku": {"name": "Basic"},
            "admin_user_enabled": False
        },
    )

    registry = poller.result()
    print(f"Registry created: {registry.login_server}")


if __name__ == "__main__":
    main()
