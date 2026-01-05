import asyncio
import sys
import os

import warnings
# Suppress specific LangGraph deprecation warning about create_react_agent
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure we can find the installed packages
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import StructuredTool

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_agent():
    # 1. Define the server parameters
    env = os.environ.copy()
    # Attempt to use TCP which is often more reliable for Python libs on Windows
    # User must have "Expose daemon on tcp://localhost:2375" enabled
    env["DOCKER_HOST"] = "tcp://localhost:2375"
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "docker-mcp"],
        env=env
    )

    print("Connecting to Docker MCP server...")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 2. Initialize connection
            await session.initialize()
            
            # 3. List available tools
            tools_response = await session.list_tools()
            mcp_tools = tools_response.tools
            print(f"Found {len(mcp_tools)} tools from MCP server.")

            # 4. Adapt MCP tools to LangChain tools
            langchain_tools = []
            for tool_def in mcp_tools:
                tool_name = tool_def.name # Capture for closure
                
                # Define sync wrapper that raises error (we are fully async)
                def sync_wrapper(**kwargs):
                    raise NotImplementedError("This tool is async-only.")

                # Define async wrapper for the actual MCP call
                # We use a default arg to capture tool_name in the closure loop
                async def async_wrapper(t_name=tool_name, **kwargs):
                    print(f"DEBUG: Calling tool '{t_name}' with args: {kwargs}")
                    res = await session.call_tool(t_name, arguments=kwargs)
                    print(f"DEBUG: Tool '{t_name}' returned.")
                    return res
                
                lc_tool = StructuredTool.from_function(
                    func=sync_wrapper,
                    coroutine=async_wrapper,
                    name=tool_def.name,
                    description=tool_def.description or "",
                )
                langchain_tools.append(lc_tool)

            # 5. Setup LLM and Agent (LangGraph)
            llm = ChatOllama(model="llama3.2:3b", temperature=0)
            
            # Create the graph-based agent
            agent_executor = create_react_agent(llm, langchain_tools)

            # 6. Run the query
            query = "List all running containers, please."
            print(f"\nTime to act! User asks: '{query}'\n")
            
            # LangGraph usage: .invoke or .ainvoke returns the state
            # debug=True prints internal steps
            response = await agent_executor.ainvoke({"messages": [("human", query)]}, debug=True)
            
            print("\nFinal Answer:")
            # The last message in 'messages' is the result
            print(response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(run_agent())
