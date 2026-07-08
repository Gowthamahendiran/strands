import os
import sys

# Ensure the current directory is in the path for importing secrets
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from secrets import get_openai_credentials
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.session.file_session_manager import FileSessionManager
from strands_tools.calculator import calculator
from strands_tools.current_time import current_time

# FileSessionManager is simply a session storage implementation that saves the conversation to files on disk.
# It is not a distributed session manager, so it will not work in a multi-process/multi-container setup.
# For a distributed session manager, see strands.session.redis_session_manager

def run_session_demo():
    # Setup custom storage directory in workspace
    storage_dir = os.path.join(current_dir, "sessions")
    session_id = "demo-session-36"
    
    # Cleanup any existing session for demonstration repeatability
    session_path = os.path.join(storage_dir, f"session_{session_id}")
    if os.path.exists(session_path):
        import shutil
        shutil.rmtree(session_path)
        print("Cleared previous session data.")
        
    # 1. Create a FileSessionManager
    session_manager = FileSessionManager(
        session_id=session_id,
        storage_dir=storage_dir
    )
        
    api_key, model_name = get_openai_credentials()
    model = OpenAIModel(
        model_id=model_name,
        client_args={"api_key": api_key}
    )
    
    print("\n--- Session Initialization ---")
    print(f"Creating Agent with Session ID: {session_id}")
    
    # Create the agent with session manager
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
    
    # 2. Simulate a restart or a new agent instance by creating a NEW session manager
    # referencing the same session_id and storage_dir.
    print("\n--- Simulating Agent Restart/New Instance ---")
    print("Creating a new Session Manager and a new Agent instance...")
    
    new_session_manager = FileSessionManager(
        session_id=session_id,
        storage_dir=storage_dir
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
    
    # Let's inspect the stored session files on disk
    print("\n--- Stored Session Files on Filesystem ---")
    session_path = os.path.join(storage_dir, f"session_{session_id}")
    for root, dirs, files in os.walk(session_path):
        for name in files:
            rel_path = os.path.relpath(os.path.join(root, name), storage_dir)
            print(f"- {rel_path}")

if __name__ == "__main__":
    run_session_demo()
