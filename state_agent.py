import os
import sys

# Ensure the current directory is in the path for importing secrets
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from secrets import get_openai_credentials
from strands import Agent, tool, ToolContext
from strands.models.openai import OpenAIModel

# ToolContext is an object that gives a tool access to the agent that is currently running.

@tool(context=True)
def update_booking_state(
    tool_context: ToolContext,
    current_step: str = None,
    selected_flight: str = None,
    selected_seat: str = None,
    payment_status: str = None,
    increment_retry: bool = False
) -> str:
    """Update the booking-related state properties in the agent state.

    Parameters:
      current_step: The current step in the booking process (e.g., 'flight_selection', 'seat_selection', 'payment', 'confirmed').
      selected_flight: The flight code selected by the user (e.g., 'AI-203').
      selected_seat: The seat number selected by the user (e.g., '12A').
      payment_status: The status of the payment (e.g., 'NotStarted', 'Pending', 'Completed', 'Failed').
      increment_retry: If True, increments the retry count by 1.

    Returns:
      A confirmation string showing the updated state.
    """
    state = tool_context.agent.state
    
    if current_step is not None:
        state.set("current_step", current_step)
    if selected_flight is not None:
        state.set("selected_flight", selected_flight)
    if selected_seat is not None:
        state.set("selected_seat", selected_seat)
    if payment_status is not None:
        state.set("payment_status", payment_status)
    if increment_retry:
        retry_count = state.get("retry_count") or 0
        state.set("retry_count", retry_count + 1)
        
    return f"Booking state updated. Current state: {state.get()}"

@tool(context=True)
def get_booking_state(tool_context: ToolContext) -> str:
    """Retrieve the current booking state from the agent state.

    Returns:
      A formatted string containing all current booking state properties.
    """
    state = tool_context.agent.state
    current_step = state.get("current_step")
    selected_flight = state.get("selected_flight")
    selected_seat = state.get("selected_seat")
    payment_status = state.get("payment_status")
    retry_count = state.get("retry_count")
    
    return (
        f"Current Booking State:\n"
        f"- Current Step: {current_step}\n"
        f"- Selected Flight: {selected_flight}\n"
        f"- Selected Seat: {selected_seat}\n"
        f"- Payment Status: {payment_status}\n"
        f"- Retry Count: {retry_count}"
    )

def create_state_agent() -> Agent:
    """Create and initialize the Strands agent with state and stateful tools."""
    api_key, model_name = get_openai_credentials()
    
    # Initialize the OpenAI model
    model = OpenAIModel(
        model_id=model_name,
        client_args={
            "api_key": api_key,
        }
    )
    
    system_prompt = (
        "You are a helpful flight booking assistant that tracks the user's booking flow in your state. "
        "The state tracks: current_step, selected_flight, selected_seat, payment_status, and retry_count. "
        "Use the update_booking_state tool to modify the state, and "
        "get_booking_state to view the state. Keep responses concise."
    )
    
    # Initialize agent state with default values
    initial_state = {
        "current_step": "flight_selection",
        "selected_flight": None,
        "selected_seat": None,
        "payment_status": "NotStarted",
        "retry_count": 0
    }
    
    # Create the agent with initial state and tools
    agent = Agent(
        model=model,
        tools=[update_booking_state, get_booking_state],
        system_prompt=system_prompt,
        state=initial_state
    )
    
    return agent

if __name__ == "__main__":
    agent = create_state_agent()
    print("Agent with state created successfully.")
    
    # Query 1: Book flight and select seat
    print("\n--- Query 1 ---")
    query_1 = "Hi, I'd like to book flight AI-203 and select seat 12A. Let's move to the payment step."
    print(f"User: {query_1}")
    response_1 = agent(query_1)
    print(f"Agent: {response_1}")
    print(f"Current State: {agent.state.get()}")
    
    # Query 2: Simulate a retry/pending status
    print("\n--- Query 2 ---")
    query_2 = "The payment failed, let's retry. Update payment status to Pending and increment retry count."
    print(f"User: {query_2}")
    response_2 = agent(query_2)
    print(f"Agent: {response_2}")
    print(f"Current State: {agent.state.get()}")
    
    # Query 3: Ask what the agent remembers
    print("\n--- Query 3 ---")
    query_3 = "What is my current booking state?"
    print(f"User: {query_3}")
    response_3 = agent(query_3)
    print(f"Agent: {response_3}")
    print(f"Current State: {agent.state.get()}")
