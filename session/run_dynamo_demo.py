import os
import sys
import boto3

# Ensure project root directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from secrets import get_openai_credentials
from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools.calculator import calculator
from strands_tools.current_time import current_time

from dynamo_persistence.setup_db import create_session_table
from dynamo_persistence.session_manager import DynamoDBSessionManager

def cleanup_session(table_name: str, session_id: str, client_id: str):
    """Directly delete the single item matching the session ID and client ID."""
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)
    try:
        table.delete_item(
            Key={
                "SessionId": session_id,
                "ClientId": client_id
            }
        )
        print(f"Cleared previous session data for session '{session_id}' (Client: '{client_id}') from DynamoDB.")
    except Exception as e:
        print(f"Error clearing previous session data: {e}")

def run_dynamo_session_demo():
    table_name = "strandssession"
    session_id = "demo-session-36"
    client_id = "twitter-5a3f939c-8e17-4343-b7ceb"

    # 1. Ensure table exists with correct HASH/RANGE schema (re-creates if key schema is wrong)
    create_session_table(table_name=table_name)

    # 2. Cleanup previous session data for repeatability
    cleanup_session(table_name, session_id, client_id)

    # 3. Create the DynamoDBSessionManager
    session_manager = DynamoDBSessionManager(
        session_id=session_id,
        table_name=table_name,
        client_id=client_id
    )

    api_key, model_name = get_openai_credentials()
    model = OpenAIModel(
        model_id=model_name,
        client_args={"api_key": api_key}
    )

    print("\n--- Session Initialization with DynamoDB (Composite Primary Key Schema) ---")
    print(f"Creating Agent with Session ID: {session_id} and Client ID: {client_id}")

    # Create the agent with DynamoDB session manager
    agent = Agent(
        model=model,
        tools=[calculator, current_time],
        system_prompt="You are a helpful assistant.",
        session_manager=session_manager
    )

    # Send a message to be remembered
    message_1 = """Help me buy a gaming laptop.

My budget is ₹90,000.
Suggest some options."""
    print(f"User: {message_1}")
    response_1 = agent(message_1)
    print(f"Agent: {response_1}")

    # 4. Simulate a restart/new agent instance by creating a NEW session manager
    # referencing the same session_id.
    print("\n--- Simulating Agent Restart/New Instance (Reconnecting to DynamoDB) ---")
    print("Creating a new DynamoDBSessionManager and a new Agent instance...")

    new_session_manager = DynamoDBSessionManager(
        session_id=session_id,
        table_name=table_name,
        client_id=client_id
    )

    new_agent = Agent(
        model=model,
        tools=[calculator, current_time],
        system_prompt="You are a helpful assistant.",
        session_manager=new_session_manager
    )

    # Query the new agent instance about the fact stated previously
    message_2 = """Continue helping me choose the laptop.
What was my budget?"""
    print(f"User: {message_2}")
    response_2 = new_agent(message_2)
    print(f"Agent: {response_2}")

    # Inspect the items stored in the DynamoDB table
    print("\n--- Stored Session Record in DynamoDB Table ---")
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)
    response = table.get_item(
        Key={
            "SessionId": session_id,
            "ClientId": client_id
        }
    )
    item = response.get("Item", {})
    
    # Print the outer keys
    for k in ["SessionId", "ClientId", "CreatedAt", "LastUpdated", "MessageCount"]:
        print(f"{k}: {item.get(k)}")
        
    print("\nHistory:")
    history = item.get("History", [])
    for idx, h_item in enumerate(history):
        print(f"[{idx}] Type: {h_item.get('type')}")
        data = h_item.get("data", {})
        print(f"    Content: {data.get('content')}")
        print(f"    Message ID: {h_item.get('message_id')}")
        print(f"    Redact Message: {h_item.get('redact_message')}")
        
        # Display the response_metadata (usage and metrics)
        response_metadata = data.get("response_metadata", {})
        if response_metadata:
            print(f"    Response Metadata:")
            print(f"        Usage: {response_metadata.get('usage')}")
            print(f"        Metrics: {response_metadata.get('metrics')}")
        print("-" * 40)

if __name__ == "__main__":
    run_dynamo_session_demo()
