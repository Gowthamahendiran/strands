import os
import sys

# Ensure the current directory is in the path for importing secrets
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from secrets import get_openai_credentials
from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools.calculator import calculator

def demo_context_management():
    api_key, model_name = get_openai_credentials()
    model = OpenAIModel(
        model_id=model_name,
        client_args={"api_key": api_key}
    )
    
    # 1. Automatic Context Management Mode
    print("\n=== 1. Automatic Context Management Mode ===")
    print("Initializing agent with context_manager='auto'...")
    auto_agent = Agent(
        model=model,
        tools=[calculator],
        system_prompt="You are a helpful assistant.",
        context_manager="auto" # Strands manage the context automatically -> The agent handles it behind the scenes.
    )
    print(f"Registered tools: {auto_agent.tool_names}")
    
    query_1 = "Calculate 3111696 / 74088."
    print(f"User: {query_1}")
    response_1 = auto_agent(query_1)
    print(f"Agent: {response_1}")
    
    # 2. Agentic Context Management Mode
    print("\n=== 2. Agentic Context Management Mode ===")
    print("Initializing agent with context_manager='agentic'...")
    agentic_agent = Agent(
        model=model,
        tools=[calculator],
        system_prompt="You are a helpful assistant.",
        context_manager="agentic" # Now the LLM itself gets tools to manage its own memory.
    )
    print(f"Registered tools: {agentic_agent.tool_names}")
    
    query_2 = "Calculate 3111696 / 74088."
    print(f"User: {query_2}")
    response_2 = agentic_agent(query_2)
    print(f"Agent: {response_2}")

if __name__ == "__main__":
    demo_context_management()
