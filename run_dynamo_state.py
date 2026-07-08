import os
import sys
import json
import boto3
from typing import Any
from strands import Agent, tool, ToolContext
from strands.models.openai import OpenAIModel

# Ensure current directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from secrets import get_openai_credentials
from dynamo_persistence.setup_db import create_session_table
from dynamo_persistence.session_manager import DynamoDBSessionManager

@tool(context=True)
def save_information_in_state(tool_context: ToolContext, key: str, value: Any) -> str:
    """Save any important information in the agent's long-term memory state.
    
    Use this to remember facts, steps, choices, or details about the user.
    """
    tool_context.agent.state.set(key, value)
    return f"Successfully saved memory: {key} = {value}"

@tool(context=True)
def get_information_from_state(tool_context: ToolContext, key: str) -> str:
    """Retrieve saved information from the agent's memory state by key."""
    val = tool_context.agent.state.get(key)
    return f"Value of '{key}': {val}"

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

def run_dynamo_state_demo():
    table_name = "strandssession"
    session_id = "state-demo-session-28"
    client_id = "twitter-5a3f939c-8e17-4343-b7ceb"
    agent_id = "booking_agent"

    # 1. Ensure table exists with correct HASH/RANGE schema
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

    system_prompt = (
        "You are a helpful flight booking assistant that tracks the user's booking flow in your state. "
        "Use the save_information_in_state tool to remember facts, details, choices, or steps (like budget, selected_flight, selected_seat, current_step). "
        "Use the get_information_from_state tool to retrieve previously saved memory state keys. Keep responses concise."
    )

    print("\n--- Session & State Initialization with DynamoDB (Generic State Tools) ---")
    print(f"Creating Agent with Session ID: {session_id}, Client ID: {client_id}, Agent ID: {agent_id}")

    # Create the agent with DynamoDB session manager and generic tools (no initial state)
    agent = Agent(
        agent_id=agent_id,
        model=model,
        tools=[save_information_in_state, get_information_from_state],
        system_prompt=system_prompt,
        session_manager=session_manager
    )

    # Query 1: Book flight, select seat, state budget
    print("\n--- Query 1 ---")
    query_1 = "Hi, I'd like to book flight AI-203 and select seat 12A. Let's move to the payment step. Also, my budget is ₹90,000."
    print(f"User: {query_1}")
    response_1 = agent(query_1)
    print(f"Agent: {response_1}")
    print(f"Current State in Memory: {agent.state.get()}")

    # 4. Simulate a restart/new agent instance by creating a NEW session manager
    # referencing the same session_id and client_id.
    print("\n--- Simulating Agent Restart/New Instance (Reconnecting to DynamoDB) ---")
    print("Creating a new DynamoDBSessionManager and a new Agent instance...")

    new_session_manager = DynamoDBSessionManager(
        session_id=session_id,
        table_name=table_name,
        client_id=client_id
    )

    new_agent = Agent(
        agent_id=agent_id,
        model=model,
        tools=[save_information_in_state, get_information_from_state],
        system_prompt=system_prompt,
        session_manager=new_session_manager
    )

    print(f"Restored Agent State in Memory immediately after creation: {new_agent.state.get()}")

    # Query 2: Ask the new agent what it remembers from the state
    print("\n--- Query 2 (Asking the new instance) ---")
    query_2 = "What flight did I book, what is my seat, and what was my budget?"
    print(f"User: {query_2}")
    response_2 = new_agent(query_2)
    print(f"Agent: {response_2}")
    print(f"Current State in Memory: {new_agent.state.get()}")

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
        
    print("\nAgentState (Persisted Agent States):")
    agent_states = item.get("AgentState", {})
    for a_id, state_json in agent_states.items():
        print(f"  Agent: {a_id}")
        state_dict = json.loads(state_json)
        print(f"  State Value: {state_dict.get('state')}")
        print(f"  Created At: {state_dict.get('created_at')}")
        print(f"  Updated At: {state_dict.get('updated_at')}")

    print("\nHistory:")
    history = item.get("History", [])
    for idx, h_item in enumerate(history):
        print(f"[{idx}] Type: {h_item.get('type')}")
        data = h_item.get("data", {})
        print(f"    Content: {data.get('content')}")
        print(f"    Message ID: {h_item.get('message_id')}")
        print(f"    Redact Message: {h_item.get('redact_message')}")
        print("-" * 40)

if __name__ == "__main__":
    run_dynamo_state_demo()
