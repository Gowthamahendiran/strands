#!/usr/bin/env python3
"""
mcp_client_test.py - A Strands Agent client wrapper for testing MCP HTTP/SSE integration.

This client runs within the 'strands' project, connects to the already-running
MCP HTTP server at 'http://localhost:8000/sse' using Server-Sent Events (SSE) transport,
and registers its tools into a Strands Agent.
"""

import os
import sys
import re
from mcp.client.sse import sse_client
from strands_secrets import get_openai_credentials
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.tools.mcp import MCPClient

# =====================================================================
# 1. SETUP ENVIRONMENT & PATHS
# =====================================================================
# Ensure the current directory is in the sys.path so we can import strands packages
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# SSE Connection URL
SERVER_URL = "http://localhost:8000/sse"

# =====================================================================
# 2. RUN STRANDS AGENT OVER HTTP/SSE
# =====================================================================
def run_agent_workflow():
    print("=" * 70)
    print("              STARTING STRANDS AGENT WITH HTTP/SSE MCP")
    print("=" * 70)
    print(f"Connecting to MCP Server at: {SERVER_URL}\n")

    # 2.1 Initialize OpenAI LLM Model
    api_key, model_name = get_openai_credentials()
    model = OpenAIModel(
        model_id=model_name,
        client_args={
            "api_key": api_key,
        }
    )

    # 2.2 Initialize the MCPClient using sse_client transport and apply tool filters
    # The client connects over the network to the already running server.
    mcp_client = MCPClient(
        transport_callable=lambda: sse_client(SERVER_URL), # Creates an SSE (HTTP) connection to the server when needed.
        tool_filters={
            "allowed": [
                re.compile(r"^test.*")
            ]
        }
    )

    # 2.3 Create the Strands Agent
    # Pass the MCPClient directly in the `tools` list.
    # Strands connects to the SSE stream, lists the tools, and registers them.
    agent = Agent(
        model=model,
        tools=[mcp_client],
        system_prompt=(
            "You are a helpful assistant with access to remote MCP tools. "
            "Use the 'test_greet_user', 'test_add_numbers', 'test_sub_numbers', and 'info_get_system_info' "
            "tools whenever necessary to satisfy the user request."
        )
    )

    unfiltered_tools = mcp_client.list_tools_sync(tool_filters={})
    print(f"[Tool Filter] Server exposed {len(unfiltered_tools)} tools.")
    if mcp_client._loaded_tools is not None:
        print(f"[Tool Filter] After applying filters, {len(mcp_client._loaded_tools)} tool(s) were registered:")
        print(f"{[t.tool_name for t in mcp_client._loaded_tools]}")
    
    # prompt = "Hello! Please greet 'Bob', add the numbers 24.5 and 75.5, and check the server's platform information."
    prompt = "HI, My name is Gowtham"
    print(f"\nPrompt: '{prompt}'")
    print("Running agent reasoning loop...\n")

    try:
        response = agent(prompt)
        print("\n" + "=" * 50)
        print(" AGENT RESPONSE")
        print("=" * 50)
        print(response)
        print("=" * 50)
    except Exception as e:
        print(f"Error during agent execution: {e}")
    finally:
        # Stop and cleanup the client's SSE transport connection
        print("\n[Client] Cleaning up MCP Client session...")
        mcp_client.stop(None, None, None)

    print("\n" + "=" * 70)
    print("              STRANDS MCP AGENT EXECUTION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    run_agent_workflow()
