import os
import sys
from typing import Any

# Ensure the current directory is in the path for importing secrets
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from secrets import get_openai_credentials
from strands import Agent, tool
from strands.models.openai import OpenAIModel
from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry

@tool
def delete_files(paths: list[str]) -> bool:
    """Delete a list of file paths.

    Parameters:
      paths: The list of files to delete.
    """
    print(f"\n[Tool Execution] Successfully deleted files: {paths}")
    return True

# This hook watches tool execution.
class ApprovalHook(HookProvider):
    def __init__(self, app_name: str) -> None:
        self.app_name = app_name

# Tool About To Execute

# ↓

# Run approve()

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self.approve)

    def approve(self, event: BeforeToolCallEvent) -> None:
        if event.tool_use["name"] != "delete_files":
            return

        # Raise an interrupt to pause execution and ask for approval
        approval = event.interrupt(
            f"{self.app_name}-approval",
            reason={"paths": event.tool_use["input"]["paths"]}
        )
        
        print(f"\n[Approval Hook] Received interrupt response: {approval}")
        
        # If response is not 'y', cancel the tool execution
        if approval.lower() != "y":
            event.cancel_tool = f"User denied permission to delete files (Response: {approval})"

def run_interrupts_demo():
    api_key, model_name = get_openai_credentials()
    model = OpenAIModel(
        model_id=model_name,
        client_args={"api_key": api_key}
    )
    
    # Initialize agent with ApprovalHook
    agent = Agent(
        model=model,
        hooks=[ApprovalHook("myapp")],
        system_prompt="You delete files as requested by the user.",
        tools=[delete_files],
        callback_handler=None
    )
    
    paths_to_delete = ["logs/system.log", "temp/cache.db"]
    query = f"Please delete the files at paths: {paths_to_delete}"
    print(f"User: {query}")
    
    # First invocation
    result = agent(query)
    
    print(f"\nStop Reason: {result.stop_reason}")
    
    # Loop to handle interrupts programmatically
    while True:
        if result.stop_reason != "interrupt":
            break
            
        responses = []
        for interrupt in result.interrupts:
            if interrupt.name == "myapp-approval":
                print(f"[Approval Request] App '{interrupt.name}' requested to delete: {interrupt.reason['paths']}")
                
                # Programmatically approving with 'y'
                user_response = "y"
                print(f"[User Approval] Programmatic Input: {user_response}")
                
                responses.append({
                    "interruptResponse": {
                        "interruptId": interrupt.id,
                        "response": user_response
                    }
                })
        
        # Resume the agent loop by invoking the agent with the response payload
        print("\nResuming agent with approval responses...")
        result = agent(responses)

    print(f"\nFinal Agent Response: {result}")

if __name__ == "__main__":
    run_interrupts_demo()
