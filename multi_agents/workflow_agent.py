import os
import sys
import logging

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
from strands.multiagent import GraphBuilder
from pinecone_memory import init_telemetry_and_logging

def create_sequential_workflow():
    # 1. Load credentials
    api_key, model_name = get_openai_credentials()
    model = OpenAIModel(
        model_id=model_name,
        client_args={
            "api_key": api_key,
        }
    )
    
    # 2. Define workflow agents
    # Agent 1: Planner creates a very brief outline.
    planner_agent = Agent(
        model=model,
        system_prompt=(
            "You are a Content Planner. Create a very brief, 3-bullet outline "
            "for a micro-blog post about the topic. Keep it under 50 words."
        ),
        name="planner_agent"
    )
    
    # Agent 2: Writer drafts a brief summary.
    writer_agent = Agent(
        model=model,
        system_prompt=(
            "You are a Technical Writer. Take the outline and write a short, punchy, "
            "one-paragraph summary of the topic. Keep it strictly under 100 words."
        ),
        name="writer_agent"
    )
    
    # Agent 3: Editor refines it to be very short.
    editor_agent = Agent(
        model=model,
        system_prompt=(
            "You are a Copy Editor. Review the summary. Refine it for maximum impact, "
            "keep it under 100 words, and format it cleanly with markdown."
        ),
        name="editor_agent"
    )
    
    # 3. Construct a sequential workflow using GraphBuilder
    # In strands, workflows are built deterministically using directed edges.
    builder = GraphBuilder()
    
    # Add agents to the workflow
    builder.add_node(planner_agent, node_id="planner")
    builder.add_node(writer_agent, node_id="writer")
    builder.add_node(editor_agent, node_id="editor")
    
    # Define linear sequential edges: planner -> writer -> editor
    builder.add_edge("planner", "writer")
    builder.add_edge("writer", "editor")
    
    # The entry point of our workflow is the planner
    builder.set_entry_point("planner")
    
    # Set safety limits to avoid infinite loops and silence warnings
    builder.set_max_node_executions(10)
    builder.set_execution_timeout(300)
    
    workflow = builder.build()
    return workflow

if __name__ == "__main__":
    print("Initializing Langfuse Tracing...")
    langfuse_client = init_telemetry_and_logging()
    
    print("Initializing Sequential Content Workflow...")
    workflow = create_sequential_workflow()
    
    topic = "The Rise of Edge Computing and its Impact on IoT Devices"
    print(f"\n--- Running Workflow on Topic: '{topic}' ---")
    try:
        result = workflow(topic)
        
        print("\nWorkflow Execution Order:")
        print(" -> ".join([node.node_id for node in result.execution_order]))
        
        # Display the output at each step of the pipeline
        print("\n==================================================")
        print("STEP 1: Planner Output:")
        print("==================================================")
        print(result.results["planner"].result)
        
        print("\n==================================================")
        print("STEP 2: Writer Output:")
        print("==================================================")
        print(result.results["writer"].result)
        
        print("\n==================================================")
        print("STEP 3: Editor (Final Publication) Output:")
        print("==================================================")
        print(result.results["editor"].result)
        print("==================================================")
        
    except Exception as e:
        print(f"Error executing workflow: {e}")

    print("\nFlushing traces to Langfuse...")
    langfuse_client.flush()
    print("Tracing completed successfully!")
