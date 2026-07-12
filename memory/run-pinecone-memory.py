import os
import sys
import base64
from strands import Agent
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

    # Create the agent with the memory manager config
    agent = Agent(
        model=model,
        system_prompt=(
            "You are a helpful assistant with access to the user's permanent memories. "
            "Use search_memory to look up user facts, and add_memory to save new facts. "
            "Always search memory first when the user asks what you know about them or asks you to recall facts."
        ),
        memory_manager=memory_manager
    )

    print("\n=== Query 1: Ask agent to remember new facts ===")
    query_1 = "Please remember that my name is Gowtham and my favorite coding language is Python. Also remember I live in Chennai."
    print(f"User: {query_1}")
    response_1 = agent(query_1)
    print(f"Agent: {response_1}\n")

    # 4. Simulate agent restart / new session
    # Creating a new agent instance, which will query the database on-demand
    print("=== Simulating Agent Restart/New Session ===")
    print("Creating a fresh Agent instance with the Pinecone Memory Store...")
    
    new_memory_store = PineconeMemoryStore(
        index_name=PINECONE_INDEX_NAME,
        client_id=client_id,
        api_key=PINECONE_API_KEY,
        openai_api_key=api_key
    )
    new_memory_manager = MemoryManager(
        stores=[new_memory_store],
        add_tool_config=True
    )
    
    new_agent = Agent(
        model=model,
        system_prompt=(
            "You are a helpful assistant with access to the user's permanent memories. "
            "Use search_memory to look up user facts, and add_memory to save new facts. "
            "Always search memory first when the user asks what you know about them or asks you to recall facts."
        ),
        memory_manager=new_memory_manager
    )

    print("\n=== Query 2: Ask the fresh instance to recall ===")
    query_2 = "What is my name, my favorite programming language, and where do I live?"
    print(f"User: {query_2}")
    response_2 = new_agent(query_2)
    print(f"Agent: {response_2}\n")

    # 5. Print stored items directly from Pinecone Index
    print("=== Stored Memory Records in Pinecone ('strands-memory' Index) ===")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    try:
        response = index.query(
            vector=[0.0] * 512,
            top_k=10,
            filter={"client_id": {"$eq": client_id}},
            include_metadata=True
        )
        matches = response.get("matches", [])
        for idx, match in enumerate(matches):
            metadata = match.get("metadata", {})
            print(f"[{idx}] ClientId: {metadata.get('client_id')}")
            print(f"    MemoryId: {match.get('id')}")
            print(f"    Content: {metadata.get('content')}")
            print(f"    Created At: {metadata.get('created_at')}")
            print(f"    Score (Similarity): {match.get('score')}")
            print("-" * 50)
    except Exception as e:
        print(f"Error querying Pinecone for inspection: {e}")

    # 6. Flush Langfuse traces to ensure all events are sent before script terminates
    print("Flushing traces to Langfuse...")
    langfuse_client.flush()

if __name__ == "__main__":
    run_pinecone_memory_demo()
