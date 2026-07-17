import os
import sys

# Ensure the parent directory is in the path for importing secrets and strands modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dotenv import load_dotenv
load_dotenv()

from strands_secrets import get_openai_credentials
from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools.graph import graph
from pinecone_memory import init_telemetry_and_logging

def run_graph_tool_demo():
    # 1. Initialize Langfuse Native Tracing
    print("Initializing Langfuse Tracing...")
    langfuse_client = init_telemetry_and_logging()

    # 2. Load LLM credentials
    api_key, model_name = get_openai_credentials()
    model = OpenAIModel(
        model_id=model_name,
        client_args={
            "api_key": api_key,
        }
    )

    # 3. Create the parent agent (coordinator)
    # The parent agent provides model and tool inheritance configuration to the graph tool.
    parent_agent = Agent(
        model=model,
        system_prompt="You are a coordinator agent supervising a multi-agent graph network.",
        name="coordinator_agent"
    )

    graph_id = "content_pipeline_via_tool"

    # 4. Define graph topology (nodes, edges, and configuration)
    print(f"\n--- Creating Graph via high-level graph tool: '{graph_id}' ---")
    topology = {
        "nodes": [
            {
                "id": "researcher",
                "role": "researcher",
                "system_prompt": (
                    "You are a Research Assistant. Research the given topic thoroughly "
                    "and output a bulleted list of the most critical facts and developments."
                )
            },
            {
                "id": "writer",
                "role": "writer",
                "system_prompt": (
                    "You are a Technical Writer. Take the research facts provided by the researcher "
                    "and write a short, cohesive one-paragraph summary. Keep it under 100 words."
                )
            }
        ],
        "edges": [
            {"from": "researcher", "to": "writer"}
        ],
        "entry_points": ["researcher"]
    }

    # Use the graph tool to compile the graph (passes the parent agent reference)
    create_result = graph(
        action="create",
        graph_id=graph_id,
        topology=topology,
        agent=parent_agent
    )
    print("Create Result:", create_result)

    # 5. Execute a task on the created graph
    task_query = "The current state of quantum computing and its practical applications."
    print(f"\n--- Executing Task on Graph: '{task_query}' ---")
    
    execution_result = graph(
        action="execute",
        graph_id=graph_id,
        task=task_query,
        agent=parent_agent
    )
    
    print("\nExecution Result Status:", execution_result.get("status"))
    if execution_result.get("status") == "success":
        print("\nFinal Executed Result:")
        for block in execution_result.get("content", []):
            if "text" in block:
                print(block["text"])

    # 6. Flush traces to Langfuse
    print("Flushing traces to Langfuse...")
    langfuse_client.flush()
    print("Done!")

if __name__ == "__main__":
    run_graph_tool_demo()
