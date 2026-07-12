import os
import sys

# Ensure the current directory is in the path for importing secrets
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from strands_secrets import get_openai_credentials
from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools.calculator import calculator
from strands_tools.current_time import current_time

def create_strands_agent() -> Agent:
    """Create and initialize the Strands agent with OpenAIModel and tools."""
    api_key, model_name = get_openai_credentials()
    
    # Initialize the OpenAI model with the retrieved credentials
    model = OpenAIModel(
        model_id=model_name,
        client_args={
            "api_key": api_key,
        }
    )
    
    system_prompt = (
        "You are a helpful assistant equipped with custom tools. "
        "Use the calculator and current_time tools "
        "to assist the user with their requests."
    )
    
    # Initialize the agent with the model, tools, and system prompt
    agent = Agent(
        model=model,
        tools=[calculator, current_time],
        system_prompt=system_prompt
    )
    
    return agent

if __name__ == "__main__":
    agent = create_strands_agent()
    print("Agent created successfully.")
    
    message = """ Guess my age. I was born on 27th May 2001. Please calculate my age in years, months, and days based on the current date. """
    
    try:
        response = agent(message)
        print(f"\nAgent response:\n{response}")
    except Exception as e:
        print(f"Error executing agent query: {e}")
