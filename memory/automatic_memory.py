import os
import sys
import asyncio
from pinecone import Pinecone
# pyrefly: ignore [missing-import]
from langfuse import Langfuse

# Ensure project root directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from secrets import get_openai_credentials, get_llm_secret
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.memory import MemoryManager
from strands.telemetry import StrandsTelemetry
from pinecone_memory.memory_store import PineconeMemoryStore

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Pinecone Configuration loaded from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "strands-memory")

def cleanup_memories(index_name: str, api_key: str, client_id: str):
    """Delete all memories for the client_id from Pinecone to ensure a clean demo run."""
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    try:
        index.delete(filter={"client_id": {"$eq": client_id}})
        print(f"Cleared previous memories for Client '{client_id}' from Pinecone.")
    except Exception as e:
        print(f"Error clearing memories: {e}")

async def run_automatic_memory_demo():
    # Load Langfuse credentials from environment variables
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL")

    # Set native Langfuse SDK environment variables
    os.environ["LANGFUSE_HOST"] = LANGFUSE_BASE_URL
    os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY

    # Initialize the Strands global OpenTelemetry TracerProvider
    StrandsTelemetry()

    # Initialize Langfuse, which hooks into the global OpenTelemetry provider
    print("Initializing Langfuse Native Tracing SDK...")
    langfuse_client = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_BASE_URL
    )

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
    print("User: Hi! My name is Gowtham.")
    resp = agent("Hi! My name is Gowtham.")
    print(f"Agent: {resp}\n")
    
    # Turn 2
    print("User: I live in Chennai.")
    resp = agent("I live in Chennai.")
    print(f"Agent: {resp}\n")
    
    # Turn 3
    print("User: My favorite coding language is Python.")
    resp = agent("My favorite coding language is Python.")
    print(f"Agent: {resp}\n")
    
    # Turn 4
    print("User: I have a pet dog named Buddy.")
    resp = agent("I have a pet dog named Buddy.")
    print(f"Agent: {resp}\n")
    
    # Turn 5 (This will hit 5 turns and trigger the extraction!)
    print("User: I love drinking filter coffee.")
    resp = agent("I love drinking filter coffee.")
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
    print("\n=== Turn 6: Querying memory context ===")
    print("User: What is my favorite programming language and where do I live?")
    resp = agent("What is my favorite programming language and where do I live?")
    print(f"Agent: {resp}\n")

    # Flush Langfuse traces before exit
    print("Flushing traces to Langfuse...")
    Langfuse().flush()

if __name__ == "__main__":
    asyncio.run(run_automatic_memory_demo())
