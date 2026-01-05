from mcp.server.fastmcp import FastMCP
from app.core import tools
from app.core.agent import memory_instance

# Initialize MCP Server
mcp = FastMCP("oppen-local-cloud")

# --- Wrapper Tools ---
# We wrap the core tool functions to expose them via MCP

@mcp.tool()
def list_pods(namespace: str = "default") -> str:
    """List Kubernetes pods in a namespace using kubectl."""
    return tools.list_pods.invoke({"namespace": namespace})

@mcp.tool()
def query_prometheus(query: str) -> str:
    """Executes a PromQL query against the cluster Prometheus."""
    return tools.query_prometheus.invoke({"query": query})

@mcp.tool()
def scale_app(app_name: str, replicas: int) -> str:
    """Scales application by updating 'terraform.tfvars.json'. ONLY supports 'podinfo'."""
    return tools.scale_app.invoke({"app_name": app_name, "replicas": replicas})

@mcp.tool()
def search_memory(query: str) -> str:
    """Searches the incident database for past similar alerts."""
    # We use the raw memory instance here since tools.get_memory_tools returns tool objects
    results = memory_instance.search_incidents(query)
    if not results:
        return "No similar past incidents found."
    import json
    return f"Found similar past incidents:\n{json.dumps(results, indent=2)}"

@mcp.tool()
def save_memory(description: str, resolution: str) -> str:
    """Saves the current incident and its resolution to memory."""
    doc_id = memory_instance.save_incident(description, resolution)
    return f"Incident saved with ID: {doc_id}"

def run_server():
    print("Starting Oppen MCP Server...")
    mcp.run()
