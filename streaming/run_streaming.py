import os
import sys
import asyncio

# Ensure project root directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from strands_secrets import get_openai_credentials
from strands import Agent
from strands.models.openai import OpenAIModel
from pinecone_memory import init_telemetry_and_logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

async def run_interactive_streaming():
    # 1. Initialize Langfuse Native Tracing
    langfuse_client = init_telemetry_and_logging()

    api_key, model_name = get_openai_credentials()
    model = OpenAIModel(
        model_id=model_name,
        client_args={"api_key": api_key}
    )

    # 2. Create the agent
    agent = Agent(
        model=model,
        system_prompt="You are a helpful assistant."
    )

    print("\n=======================================================")
    print("      Interactive Strands Streaming Demo (CLI)")
    print("=======================================================")
    print("Type your questions below. Type 'exit' to quit.")
    print("=======================================================")

    loop = asyncio.get_event_loop()

    while True:
        try:
            # Read user input asynchronously from terminal
            user_input = await loop.run_in_executor(None, lambda: input("\nUser > "))
            
            if not user_input.strip():
                continue
            if user_input.lower() == "exit":
                print("Exiting...")
                break

            print("Agent > ", end="", flush=True)

            # 3. Stream agent response events
            async for event in agent.stream_async(user_input):
                if "data" in event:
                    print(event["data"], end="", flush=True)

            print()

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

    # 4. Flush Langfuse traces on exit
    print("\nFlushing traces to Langfuse...")
    langfuse_client.flush()
    print("Done!")

if __name__ == "__main__":
    asyncio.run(run_interactive_streaming())
