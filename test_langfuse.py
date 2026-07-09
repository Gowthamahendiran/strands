import os
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langfuse import Langfuse

# Load environment variables
load_dotenv()

print("LANGFUSE_PUBLIC_KEY =", os.getenv("LANGFUSE_PUBLIC_KEY"))
print("LANGFUSE_HOST =", os.getenv("LANGFUSE_BASE_URL"))

client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL"),
)

print("Creating trace observation...")
with client.start_as_current_observation(name="test-trace") as trace:
    print("Inside trace observation")

print("Flushing client...")
client.flush()

print("Done")
