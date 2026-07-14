import os
import sys
import base64
from strands import Agent, tool
from strands.models.openai import OpenAIModel
from strands.memory import MemoryManager
from pinecone import Pinecone

# Ensure project root directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from strands_secrets import get_openai_credentials, get_llm_secret
from pinecone_memory import PineconeMemoryStore, cleanup_memories, init_telemetry_and_logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Pinecone Configuration loaded from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "strands-memory")

# --- Simple Mock Tools ---

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.

    Parameters:
      location: The name of the city or location.
    """
    return f"The weather in {location} is currently 28°C and sunny."

@tool
def get_time_of_day() -> str:
    """Get the current time of day."""
    return "The current time is 4:15 PM."

@tool
def search_internet(query: str) -> str:
    """Search the internet for info on a query.

    Parameters:
      query: The search query.
    """
    return f"Search results for '{query}': Found 3 articles discussing the topic."

@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email to a recipient.

    Parameters:
      recipient: Email address of the recipient.
      subject: Subject of the email.
      body: Body content of the email.
    """
    return f"Email with subject '{subject}' successfully sent to {recipient}."

@tool
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate Body Mass Index (BMI).

    Parameters:
      weight_kg: Weight in kilograms.
      height_m: Height in meters.
    """
    if height_m <= 0:
        return 0.0
    return round(weight_kg / (height_m ** 2), 2)

@tool
def recommend_restaurant(cuisine: str) -> str:
    """Recommend a restaurant based on cuisine type.

    Parameters:
      cuisine: The type of cuisine (e.g. Italian, Indian, Chinese).
    """
    return f"Recommended {cuisine} restaurant: 'The Tasty Plate' (Rating: 4.8/5.0)"

# --- Demo Setup ---

def run_pinecone_memory_demo():
    langfuse_client = init_telemetry_and_logging()

    client_id = "twitter-5a3f939c-8e17-4343-b7ceb"

    # 1. Cleanup previous memories in Pinecone index for repeatability
    cleanup_memories(index_name=PINECONE_INDEX_NAME, api_key=PINECONE_API_KEY, client_id=client_id)

    api_key, model_name = get_openai_credentials()
    
    # Initialize standard model client (it will be auto-traced via OpenTelemetry + Langfuse)
    model = OpenAIModel(
        model_id=model_name,
        client_args={"api_key": api_key}
    )

    # 2. Instantiate the custom Pinecone Memory Store
    memory_store = PineconeMemoryStore(
        index_name=PINECONE_INDEX_NAME,
        client_id=client_id,
        api_key=PINECONE_API_KEY,
        openai_api_key=api_key
    )

    # 3. Bind the store to the Strands MemoryManager
    memory_manager = MemoryManager(
        stores=[memory_store],
        add_tool_config=True  # Exposes add_memory and search_memory tools to the agent
    )

    # List of all tools to attach to the agents
    custom_tools = [
        get_weather,
        get_time_of_day,
        search_internet,
        send_email,
        calculate_bmi,
        recommend_restaurant
    ]

    # Create the agent with the memory manager config and tools
    agent = Agent(
        model=model,
        tools=custom_tools,
        system_prompt=(
            "You are a helpful assistant"
        ),
        memory_manager=memory_manager
    )

    # print("\n=== Query 1: Ask agent to remember new facts ===")
    query_1 = "What is AITrail and What is AI Agent"
    print(f"User: {query_1}")
    response_1 = agent(query_1)
    print(f"Agent: {response_1}\n")

    # 4. Simulate agent restart / new session
    # Creating a new agent instance, which will query the database on-demand
    # print("=== Simulating Agent Restart/New Session ===")
    # print("Creating a fresh Agent instance with the Pinecone Memory Store...")
    
    # new_memory_store = PineconeMemoryStore(
    #     index_name=PINECONE_INDEX_NAME,
    #     client_id=client_id,
    #     api_key=PINECONE_API_KEY,
    #     openai_api_key=api_key
    # )
    # new_memory_manager = MemoryManager(
    #     stores=[new_memory_store],
    #     add_tool_config=True
    # )
    
    # new_agent = Agent(
    #     model=model,
    #     tools=custom_tools,
    #     system_prompt=(
    #         "You are a helpful assistant"
    #     ),
    #     memory_manager=new_memory_manager
    # )

    # print("\n=== Query 2: Ask the fresh instance to recall ===")
    # query_2 = "What is my name, my favorite programming language, and where do I live?"
    # print(f"User: {query_2}")
    # response_2 = new_agent(query_2)
    # print(f"Agent: {response_2}\n")

    # 5. Flush Langfuse traces to ensure all events are sent before script terminates
    print("Flushing traces to Langfuse...")
    langfuse_client.flush()

if __name__ == "__main__":
    run_pinecone_memory_demo()
