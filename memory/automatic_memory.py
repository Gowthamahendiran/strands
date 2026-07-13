import os
import sys
import asyncio
from pinecone import Pinecone

# Ensure project root directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from strands_secrets import get_openai_credentials, get_llm_secret
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.memory import MemoryManager
from pinecone_memory import PineconeMemoryStore, cleanup_memories, init_telemetry_and_logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Pinecone Configuration loaded from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "strands-memory")

async def run_automatic_memory_demo():
    langfuse_client = init_telemetry_and_logging()

    client_id = "twitter-automatic-memory-demo"

    # 1. Cleanup previous memories in Pinecone index for repeatability
    cleanup_memories(index_name=PINECONE_INDEX_NAME, api_key=PINECONE_API_KEY, client_id=client_id)

    api_key, model_name = get_openai_credentials()
    
    # Initialize standard model client
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
    
    # Enable automatic memory extraction on the store
    memory_store.extraction = True

    # 3. Bind the store to the Strands MemoryManager
    memory_manager = MemoryManager(
        stores=[memory_store],
        add_tool_config=False  # No manual tools needed, extraction is automatic!
    )

    # Create the agent with the memory manager config
    agent = Agent(
        model=model,
        system_prompt="You are a helpful assistant.",
        memory_manager=memory_manager
    )

    print("\n=== Conversing with Agent (Automatic extraction will trigger every 5 turns) ===")
    
    # Turn 1
    resp = agent("Hi! My name is brother name is Mahi.")
    print(f"Agent: {resp}\n")
    
    # Turn 2
    resp = agent("I prefer dark mode.")
    print(f"Agent: {resp}\n")
    
    # Turn 3
    resp = agent("I only eat vegetarian food.")
    print(f"Agent: {resp}\n")
    
    # Turn 4
    resp = agent("I love to watch movies.")
    print(f"Agent: {resp}\n")
    
    # Turn 5 (This will hit 5 turns and trigger the extraction!)
    resp = agent("I love eating oats in morning.")
    print(f"Agent: {resp}\n")

    # Await background processes to complete extraction
    print("Flushing background extraction...")
    await memory_manager.flush()

    print("\n=== Stored Memory Records in Pinecone ===")
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
            print(f"[{idx}] Content: {metadata.get('content')}")
            print("-" * 50)
    except Exception as e:
        print(f"Error querying Pinecone: {e}")

    # Turn 6: Verify memory is remembered and queried
    resp = agent("What is my name, my preferences (like theme, food, movies), and what do I like to drink?")
    print(f"Agent: {resp}\n")


    # Flush Langfuse traces before exit
    print("Flushing traces to Langfuse...")
    langfuse_client.flush()

if __name__ == "__main__":
    asyncio.run(run_automatic_memory_demo())
