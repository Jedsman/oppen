import asyncio
import sys
import os
import json
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("terraform-server")

# Configuration
TF_DIR = os.path.join(os.getcwd(), "terraform")

async def run_terraform_async(args):
    """Helper to run terraform commands asynchronously"""
    try:
        # Check if directory exists
        if not os.path.exists(TF_DIR):
            return f"Error: Terraform directory {TF_DIR} not found."

        # Find executable - rely on PATH or assume terraform
        process = await asyncio.create_subprocess_exec(
            "terraform",
            *args,
            cwd=TF_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        output = stdout.decode().strip()
        if stderr:
             output += "\nStderr: " + stderr.decode().strip()
             
        return output
    except FileNotFoundError:
        return "Error: terraform binary not found. Is it in your PATH?"
    except Exception as e:
        return f"Error executing terraform: {str(e)}"

@mcp.tool()
async def tf_version() -> str:
    """Get the terraform version"""
    return await run_terraform_async(["version"])

@mcp.tool()
async def tf_init() -> str:
    """Initialize the terraform configuration"""
    return await run_terraform_async(["init"])

@mcp.tool()
async def tf_plan() -> str:
    """Run terraform plan execution"""
    return await run_terraform_async(["plan", "-no-color"])

@mcp.tool()
async def tf_show() -> str:
    """Show the current state"""
    return await run_terraform_async(["show", "-no-color"])

@mcp.tool()
async def tf_apply_approve() -> str:
    """Run terraform apply with auto-approve. USE WITH CAUTION."""
    return await run_terraform_async(["apply", "-auto-approve", "-no-color"])

@mcp.tool()
async def tf_validate() -> str:
    """Validate the configuration"""
    return await run_terraform_async(["validate", "-json"])

if __name__ == "__main__":
    mcp.run()
