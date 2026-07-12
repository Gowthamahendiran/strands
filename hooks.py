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
from strands.hooks import (
    HookProvider,
    HookRegistry,
    BeforeInvocationEvent,
    AfterInvocationEvent,
    BeforeToolCallEvent,
    AfterToolCallEvent,
)

class RequestLogger(HookProvider):
    """Custom Request Logger implementing the HookProvider protocol."""
    
    # When this event happens, call this function.
    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeInvocationEvent, self.log_start)
        registry.add_callback(AfterInvocationEvent, self.log_end)
        registry.add_callback(BeforeToolCallEvent, self.log_tool_use)
        registry.add_callback(AfterToolCallEvent, self.log_tool_result)

    def log_start(self, event: BeforeInvocationEvent) -> None:
        print("\n>>> [Hook: BeforeInvocationEvent] Request started!")
        print(f"    Agent ID: {event.agent.agent_id}")

    def log_end(self, event: AfterInvocationEvent) -> None:
        print(">>> [Hook: AfterInvocationEvent] Request completed!")
        print(f"    Stop Reason: {event.result.stop_reason if event.result else 'N/A'}\n")

    def log_tool_use(self, event: BeforeToolCallEvent) -> None:
        print(f"\n>>> [Hook: BeforeToolCallEvent] Intercepting tool call!")
        print(f"    Tool: {event.tool_use.get('name')}")
        print(f"    Input: {event.tool_use.get('input')}")

    def log_tool_result(self, event: AfterToolCallEvent) -> None:
        print(f">>> [Hook: AfterToolCallEvent] Tool execution complete!")
        print(f"    Tool: {event.tool_use.get('name')}")
        print(f"    Result: {event.result.get('content')}")

if __name__ == "__main__":
    api_key, model_name = get_openai_credentials()
    model = OpenAIModel(
        model_id=model_name,
        client_args={"api_key": api_key}
    )
    
    # Initialize the agent with the custom hook provider
    agent = Agent(
        model=model,
        tools=[calculator],
        system_prompt="You are a helpful assistant.",
        hooks=[RequestLogger()]
    )
    
    query = "Calculate 3111696 / 74088 and tell me what the answer is."
    print(f"User: {query}")
    
    try:
        response = agent(query)
        print(f"Agent: {response}")
    except Exception as e:
        print(f"Error: {e}")
