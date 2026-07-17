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
from strands.multiagent.graph import GraphState
from pinecone_memory import init_telemetry_and_logging

def create_graph_agent_system():
    # 1. Load credentials
    api_key, model_name = get_openai_credentials()
    model = OpenAIModel(
        model_id=model_name,
        client_args={
            "api_key": api_key,
        }
    )
    
    # 2. Define the individual specialized agents
    # "MATH - The task is a math query."
    router_agent = Agent(
        model=model,
        system_prompt=(
            "You are a routing agent. Analyze the user's task and classify it "
            "into exactly one of two categories: 'MATH' or 'GENERAL'. "
            "Respond ONLY with the category name ('MATH' or 'GENERAL') and a brief explanation."
        ),
        name="router_agent"
    )
    
    math_agent = Agent(
        model=model,
        system_prompt=(
            "You are a mathematical specialist. Solve the user's request. "
            "Provide step-by-step mathematical reasoning."
        ),
        name="math_agent"
    )
    
    general_agent = Agent(
        model=model,
        system_prompt=(
            "You are a general knowledge assistant. Answer the user's request "
            "accurately and concisely."
        ),
        name="general_agent"
    )
    
    synthesizer_agent = Agent(
        model=model,
        system_prompt=(
            "You are a response synthesizer. Take the outputs from previous agents "
            "and format them into a beautiful, cohesive final response for the user. "
            "Clearly mention which specialist agent was used to answer the query."
        ),
        name="synthesizer_agent"
    )
    
    # 3. Build the graph orchestration
    builder = GraphBuilder()
    
    # Add agents as nodes
    builder.add_node(router_agent, node_id="router")
    builder.add_node(math_agent, node_id="math_specialist")
    builder.add_node(general_agent, node_id="general_assistant")
    builder.add_node(synthesizer_agent, node_id="synthesizer")
    
    # Define routing condition functions
    def is_math_query(state: GraphState) -> bool:
        router_output = str(state.results["router"].result)
        return "MATH" in router_output.upper()
        
    def is_general_query(state: GraphState) -> bool:
        router_output = str(state.results["router"].result)
        return "GENERAL" in router_output.upper() or not is_math_query(state)
    
    # Set up edges
    builder.add_edge("router", "math_specialist", condition=is_math_query)
    builder.add_edge("router", "general_assistant", condition=is_general_query)
    builder.add_edge("math_specialist", "synthesizer")
    builder.add_edge("general_assistant", "synthesizer")
    
    # Set the start node
    builder.set_entry_point("router")
    
    # Set safety limits to avoid infinite loops and silence warnings
    builder.set_max_node_executions(10)
    builder.set_execution_timeout(300)
    
    # Build the graph
    graph = builder.build()
    return graph

if __name__ == "__main__":
    print("Initializing Langfuse Tracing...")
    langfuse_client = init_telemetry_and_logging()
    
    print("Initializing Graph Multi-Agent System...")
    graph = create_graph_agent_system()
    
    # Example 1: Math Query
    math_query = "What is the sum of the first 10 prime numbers?"
    print(f"\n--- Running Graph with Math Query: '{math_query}' ---")
    try:
        result = graph(math_query)
        print("\nFinal Synthesized Result:")
        # Find the synthesizer result in the graph results
        print(result.results["synthesizer"].result)
    except Exception as e:
        print(f"Error executing graph: {e}")

    # Example 2: General Query
    general_query = "Who wrote the play Romeo and Juliet?"
    print(f"\n--- Running Graph with General Query: '{general_query}' ---")
    try:
        result = graph(general_query)
        print("\nFinal Synthesized Result:")
        print(result.results["synthesizer"].result)
    except Exception as e:
        print(f"Error executing graph: {e}")

    print("\nFlushing traces to Langfuse...")
    langfuse_client.flush()
    print("Tracing completed successfully!")
