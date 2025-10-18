from azure.identity import DefaultAzureCredential
from azure.mgmt.containerregistry import ContainerRegistryManagementClient

subscription_id = "eb6bd382-b9e4-4511-8926-9a231a97f909"
resource_group = "rag-ai-foundations-demo-rg"
registry_name = "rag-ai-foundations-demo-registry"  # must be globally unique
location = "eastus"

# Authenticate (requires `az login` or managed identity)
credential = DefaultAzureCredential()
client = ContainerRegistryManagementClient(credential, subscription_id)

# Create registry
poller = client.registries.begin_create(
    resource_group_name=resource_group,
    registry_name=registry_name,
    registry={
        "location": location,
        "sku": {"name": "Basic"},
        "admin_user_enabled": True
    }
)

registry = poller.result()
print(f"Registry created: {registry.login_server}")
