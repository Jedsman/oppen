import typer
import asyncio
from app.interfaces.http import run_server as run_http_server
from app.interfaces.mcp import run_server as run_mcp_server
from app.core.agent import get_agent

app = typer.Typer(help="Oppen Local Cloud Agent CLI")

@app.command()
def http(port: int = 8090):
    """Start the Webhook Agent Server (FastAPI)."""
    run_http_server(port)

@app.command()
def mcp():
    """Start the Model Context Protocol Server."""
    run_mcp_server()

@app.command()
def repl():
    """Run the Agent in interactive REPL mode."""
    print("Starting Interactive Agent REPL...")
    agent = get_agent()
    
    async def run_loop():
        print("Agent Ready. Type 'exit' to quit.")
        while True:
            try:
                user_input = input("User> ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                if not user_input.strip():
                    continue
                
                print("Thinking...")
                async for chunk in agent.astream({"messages": [("human", user_input)]}, stream_mode="updates"):
                    for node, values in chunk.items():
                        if "messages" in values:
                            for msg in values["messages"]:
                                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        print(f"[Agent] Calling: {tc['name']}(...)")
                                if msg.type == 'tool':
                                    print(f"[Tool Output] {msg.content[:100]}...")
                                if msg.type == 'ai':
                                    print(f"[AI] {msg.content}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error: {e}")

    asyncio.run(run_loop())

if __name__ == "__main__":
    app()
