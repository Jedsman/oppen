import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from app.core.agent import get_agent

agent_executor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor
    print("Initializing Agent Executor...")
    agent_executor = get_agent()
    print("Agent Ready!")
    yield
    print("Shutting down...")

app = FastAPI(title="Oppen Agent Server", lifespan=lifespan)

class WebhookPayload(BaseModel):
    alert: str
    context: dict = {}

@app.post("/webhook")
async def webhook_handler(payload: WebhookPayload):
    """
    Receives an alert and triggers the Agent.
    """
    global agent_executor
    if not agent_executor:
        return {"status": "error", "message": "Agent not initialized"}

    print(f"\n[WEBHOOK] Received Alert: {payload.alert}")
    
    prompt = f"ALERT RECEIVED: {payload.alert}. Investigate the cluster state and take necessary remediation actions using your tools."
    response_text = ""
    print("[Agent] Starting investigation...")
    
    try:
        async for chunk in agent_executor.astream({"messages": [("human", prompt)]}, stream_mode="updates"):
            for node, values in chunk.items():
                if "messages" in values:
                    for msg in values["messages"]:
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tc in msg.tool_calls:
                                log = f"[Agent] Calling: {tc['name']}({tc['args']})"
                                print(log)
                                response_text += log + "\n"
                        if msg.type == 'tool':
                            content = msg.content
                            log = f"[Tool Output] {content[:200]}..."
                            print(log)
                            response_text += log + "\n"
                        if msg.type == 'ai' and not msg.tool_calls:
                             print(f"[Agent] {msg.content}")
                             response_text += f"[Answer] {msg.content}\n"
                             
        return {"status": "success", "action_log": response_text}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

def run_server(port: int = 8090):
    print(f"Starting Oppen HTTP Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
