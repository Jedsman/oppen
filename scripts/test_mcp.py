import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_test():
    
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "docker-mcp"],
        env=None
    )

    print("Connecting to Docker MCP server...")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("Calling list-containers...")
            result = await session.call_tool("list-containers", arguments={})
            print("Result received!")
            print(str(result)[:500]) # Print first 500 chars

if __name__ == "__main__":
    asyncio.run(run_test())
