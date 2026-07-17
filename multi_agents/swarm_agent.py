import os
import sys
import logging

# Ensure the parent directory is in the path for importing secrets and strands modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

from strands_secrets import get_openai_credentials
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.multiagent import Swarm

def create_customer_support_swarm():
    # 1. Load credentials
    api_key, model_name = get_openai_credentials()
    model = OpenAIModel(
        model_id=model_name,
        client_args={
            "api_key": api_key,
        }
    )
    
    # 2. Define Swarm Agents
    # Triage Agent inspects the task and decides which specialized agent should handle it.
    triage_agent = Agent(
        model=model,
        system_prompt=(
            "You are a Triage Agent. Your task is to analyze the customer query. "
            "If the query is about billing, payments, or refunds, call the handoff_to_agent tool "
            "to transfer to 'billing_agent'. "
            "If the query is technical (e.g. software issues, APIs, bugs), call the handoff_to_agent tool "
            "to transfer to 'tech_support_agent'. "
            "For all other inquiries, answer the customer directly."
        ),
        name="triage_agent"
    )
    
    # Billing Agent handles billing queries.
    billing_agent = Agent(
        model=model,
        system_prompt=(
            "You are a Billing Agent. You handle all customer inquiries about payments, refunds, and billing issues. "
            "Provide a friendly response addressing their concern. Once you are done, answer directly (do not call handoff)."
        ),
        name="billing_agent"
    )
    
    # Tech Support Agent handles technical issues.
    tech_support_agent = Agent(
        model=model,
        system_prompt=(
            "You are a Tech Support Agent. You resolve technical bugs, coding questions, and system issues. "
            "Provide technical guidance or debug details. Once you are done, answer directly (do not call handoff)."
        ),
        name="tech_support_agent"
    )
    
    # 3. Create the Swarm
    # The swarm will automatically inject the `handoff_to_agent` tool into all agents.
    support_swarm = Swarm(
        nodes=[triage_agent, billing_agent, tech_support_agent],
        entry_point=triage_agent
    )
    
    return support_swarm

if __name__ == "__main__":
    print("Initializing Support Swarm...")
    swarm = create_customer_support_swarm()
    
    # Case 1: Billing Query
    billing_query = "Hi, I was charged twice for my subscription this month. Can I get a refund for the extra charge?"
    print(f"\n--- Running Swarm with Billing Query: '{billing_query}' ---")
    try:
        result = swarm(billing_query)
        print("\nNode History (Agents Executed):")
        print(" -> ".join([node.node_id for node in result.node_history]))
        
        # Display the final agent output
        final_node_id = result.node_history[-1].node_id
        print(f"\nFinal Response (from {final_node_id}):")
        print(result.results[final_node_id].result)
    except Exception as e:
        print(f"Error executing swarm: {e}")

    # Case 2: Technical Query
    tech_query = "My script is getting a connection timeout error when sending requests to the API endpoints."
    print(f"\n--- Running Swarm with Technical Query: '{tech_query}' ---")
    try:
        result = swarm(tech_query)
        print("\nNode History (Agents Executed):")
        print(" -> ".join([node.node_id for node in result.node_history]))
        
        final_node_id = result.node_history[-1].node_id
        print(f"\nFinal Response (from {final_node_id}):")
        print(result.results[final_node_id].result)
    except Exception as e:
        print(f"Error executing swarm: {e}")
