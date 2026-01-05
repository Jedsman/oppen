import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_test():
    # We call the script using 'uv run scripts/mcp_terraform.py'
    # FastMCP uses the script execution directly
    
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "scripts/mcp_terraform.py"],
        env=os.environ.copy()
    )

    print("Connecting to Terraform MCP server...")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("Listing tools...")
            tools = await session.list_tools()
            for t in tools.tools:
                print(f"- {t.name}: {t.description}")
            
            print("\nCalling tf_version...")
            result = await session.call_tool("tf_version", arguments={})
            print("Result received:")
            print(result)

if __name__ == "__main__":
    asyncio.run(run_test())
