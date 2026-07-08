import os
import sys
import boto3
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.memory import MemoryManager

# Ensure current directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from secrets import get_openai_credentials
from dynamo_memory.setup_db import create_memory_table
from dynamo_memory.memory_store import DynamoDBMemoryStore



PineconeAPI = "pcsk_5gsq6z_7dfjfUBNPUhuedF2S6pNfoCEUqGjxpaEdr8eaCUsiEhHcpvByFx8PNy2ZE2MQQN"
def cleanup_memories(table_name: str, client_id: str):
    """Directly delete all memories for the client_id to ensure a clean demo run."""
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)
    try:
        response = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("ClientId").eq(client_id)
        )
        items = response.get("Items", [])
        for item in items:
            table.delete_item(
                Key={
                    "ClientId": client_id,
                    "MemoryId": item["MemoryId"]
                }
            )
        if items:
            print(f"Cleared {len(items)} previous memories for Client '{client_id}' from DynamoDB.")
        else:
            print(f"No previous memories found for Client '{client_id}'.")
    except Exception as e:
        print(f"Error clearing previous memories: {e}")

def run_dynamo_memory_demo():
    table_name = "strandsmemory"
    client_id = "twitter-5a3f939c-8e17-4343-b7ceb"

    # 1. Setup the dedicated memory table
    create_memory_table(table_name=table_name)

    # 2. Cleanup previous user memories for repeatability
    cleanup_memories(table_name, client_id)

    api_key, model_name = get_openai_credentials()
    model = OpenAIModel(
        model_id=model_name,
        client_args={"api_key": api_key}
    )

    # 3. Instantiate the custom DynamoDB Memory Store (imported from package)
    memory_store = DynamoDBMemoryStore(
        table_name=table_name,
        client_id=client_id,
        api_key=api_key
    )

    # 4. Bind the store to the Strands MemoryManager
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

    # 5. Simulate agent restart / new session
    # Creating a new agent instance, which will query the database on-demand
    print("=== Simulating Agent Restart/New Session ===")
    print("Creating a fresh Agent instance with the DynamoDB Memory Store...")
    
    new_memory_store = DynamoDBMemoryStore(
        table_name=table_name,
        client_id=client_id,
        api_key=api_key
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

    # 6. Print stored items directly from DynamoDB Memory Table
    print("=== Stored Memory Records in DynamoDB ('strandsmemory' table) ===")
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)
    response = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key("ClientId").eq(client_id)
    )
    items = response.get("Items", [])
    for idx, item in enumerate(items):
        print(f"[{idx}] ClientId: {item.get('ClientId')}")
        print(f"    MemoryId: {item.get('MemoryId')}")
        print(f"    Content: {item.get('content')}")
        print(f"    Created At: {item.get('created_at')}")
        # Print a preview of the embedding list to save space
        emb = item.get('embedding', [])
        print(f"    Embedding vector preview: {[float(val) for val in emb[:3]]} ... (dimensions: {len(emb)})")
        print("-" * 50)

if __name__ == "__main__":
    run_dynamo_memory_demo()
