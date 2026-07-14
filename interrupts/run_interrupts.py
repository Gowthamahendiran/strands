import os
import sys
import asyncio
from typing import Any
import requests
# Ensure project root directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from strands_secrets import get_openai_credentials
from strands import Agent, tool
from strands.models.openai import OpenAIModel
from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry
from pinecone_memory import init_telemetry_and_logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

@tool
def get_weather(location: str) -> str:
    """Get the weather forecast for a given location/city.

    Parameters:
      location: The city or location to check weather for.
    """
    print(f"\n[Tool Execution] Checking weather for: {location}...")
    # Mocking rainy weather to trigger the email condition
    return f"Tomorrow's forecast for {location}: Rainy morning with high chances of precipitation, 24°C."

@tool
def send_email(recipient: str, subject: str, body: str) -> bool:
    """Send an email to a recipient with a subject and body.

    Parameters:
      recipient: The recipient's email address.
      subject: The subject line of the email.
      body: The message body of the email.
    """
    print("\n=======================================================")
    print("            SENDING EMAIL (SIMULATED DUMMY)")
    print("=======================================================")
    print(f"To:      {recipient}")
    print(f"Subject: {subject}")
    print(f"Body:\n{body}")
    print("=======================================================")
    return True

# This hook watches tool execution and intercepts email sending to ask for approval
class ApprovalHook(HookProvider):
    def __init__(self, app_name: str) -> None:
        self.app_name = app_name

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self.approve)

    def approve(self, event: BeforeToolCallEvent) -> None:
        if event.tool_use["name"] != "send_email":
            return

        # Raise an interrupt to pause execution and ask for approval
        recipient = event.tool_use["input"]["recipient"]
        subject = event.tool_use["input"]["subject"]
        body = event.tool_use["input"]["body"]

        approval = event.interrupt(
            f"{self.app_name}-email-approval",
            reason={
                "recipient": recipient,
                "subject": subject,
                "body": body
            }
        )
        
        print(f"\n[Approval Hook] Received interrupt response: {approval}")
        
        # Use exact matching to prevent unsafe false positives (like "don't send")
        has_approved = approval.strip().lower() in {
            "y",
            "yes",
            "approve",
            "approved",
            "please send a mail",
            "go ahead",
            "do it",
            "sure",
            "ok",
            "okay",
            "proceed"
        }

        if not has_approved:
            event.cancel_tool = f"User denied permission to send email to {recipient} (Response: {approval})"

def run_interactive_interrupts():
    # 1. Initialize Langfuse Native Tracing
    langfuse_client = init_telemetry_and_logging()

    api_key, model_name = get_openai_credentials()
    model = OpenAIModel(
        model_id=model_name,
        client_args={"api_key": api_key}
    )

    # 2. Create the agent with Tools and ApprovalHook (No Session Manager)
    agent = Agent(
        model=model,
        hooks=[ApprovalHook("interactive-app")],
        system_prompt=(
            "You are a helpful assistant. You have access to weather and email tools.\n"
            "If a user asks to send an email with conditional weather info, first query the weather for their location/city "
            "(if not specified, default to checking weather for Chennai). "
            "Then, write the email and include the weather conditional information in the email body if appropriate, and call send_email."
        ),
        tools=[get_weather, send_email]
    )
    print("Type your questions below. Type 'exit' to quit.")
    print("Example: 'Send a mail to gowthamahendiran@gmail.com that there is a meeting tomorrow morning. If rain comes please bring the umbrella.'")
    print("=======================================================")

    while True:
        try:
            user_input = input("\nUser > ")
            if not user_input.strip():
                continue
            if user_input.lower() == "exit":
                print("Exiting...")
                break

            # Invoke agent with user's input
            result = agent(user_input)

            # Handle interrupts loop
            while True:
                if result.stop_reason != "interrupt":
                    break

                responses = []
                for interrupt in result.interrupts:
                    if interrupt.name == "interactive-app-email-approval":
                        print(f"\n[Interrupt Prompt] App '{interrupt.name}' wants to send an email:")
                        print(f"  To:      {interrupt.reason['recipient']}")
                        print(f"  Subject: {interrupt.reason['subject']}")
                        print(f"  Body:\n{interrupt.reason['body']}")
                        
                        # Prompt the user for input in the terminal
                        user_response = input("\nApprove sending email? (y/n): ").strip()
                        
                        responses.append({
                            "interruptResponse": {
                                "interruptId": interrupt.id,
                                "response": user_response
                            }
                        })
                
                # Resume agent loop with approval responses
                print("\nResuming agent with your decision...")
                result = agent(responses)

            print(f"\nAgent > {result}")

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
    run_interactive_interrupts()
