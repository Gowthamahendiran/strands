import os
import logging
from pinecone import Pinecone
# pyrefly: ignore [missing-import]
from langfuse import Langfuse
from strands.telemetry import StrandsTelemetry

def cleanup_memories(index_name: str, api_key: str, client_id: str):
    """Delete all memories for the client_id from Pinecone to ensure a clean demo run."""
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    try:
        # Try metadata filter delete
        index.delete(filter={"client_id": {"$eq": client_id}})
        print(f"Cleared previous memories for Client '{client_id}' from Pinecone.")
    except Exception as e:
        print(f"Error clearing memories via filter delete: {e}. Trying query-and-delete fallback...")
        try:
            # Fallback: Query all matches first and delete them by ID
            q_resp = index.query(
                vector=[0.0] * 512,
                top_k=100,
                filter={"client_id": {"$eq": client_id}},
                include_metadata=False
            )
            ids = [match["id"] for match in q_resp.get("matches", [])]
            if ids:
                index.delete(ids=ids)
                print(f"Cleared {len(ids)} previous memories for Client '{client_id}' from Pinecone.")
            else:
                print(f"No previous memories found for Client '{client_id}'.")
        except Exception as ex:
            print(f"Fallback deletion failed: {ex}")


def init_telemetry_and_logging() -> Langfuse:
    """Initialize logging configuration, Langfuse environment variables,

    and Strands OpenTelemetry TracerProvider, returning the native Langfuse client.
    """
    logging.basicConfig(level=logging.INFO)

    # Load Langfuse credentials from environment variables
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL")

    # Set native Langfuse SDK environment variables (host must be set to LANGFUSE_HOST)
    os.environ["LANGFUSE_HOST"] = LANGFUSE_BASE_URL
    os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY

    # Initialize the Strands global OpenTelemetry TracerProvider
    StrandsTelemetry()

    # Initialize Langfuse, which hooks into the global OpenTelemetry provider
    print("Initializing Langfuse Native Tracing SDK...")
    langfuse_client = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_BASE_URL
    )
    return langfuse_client
