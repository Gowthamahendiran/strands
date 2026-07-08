import os
import sys

# Ensure the current directory is in the path for importing secrets
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from secrets import get_openai_credentials
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.memory import MemoryManager
from strands.memory.types import MemoryEntry, SearchOptions

import re
import openai

# InMemoryStore is a simple in-memory store for user preferences and stable facts.
class InMemoryStore:
    name = "preferences" # Just the store name.
    description = "User preferences and stable facts."
    max_search_results = 3
    writable = True
    extraction = None

    def __init__(self) -> None:
        self._entries: list[str] = [] # Stores all memories.
        self._embeddings: dict[str, list[float]] = {} # Stores vector representations of every memory.
        self._client = None # Initially no OpenAI Client

    def _get_client(self):
        if self._client is None:
            try:
                api_key, _ = get_openai_credentials()
                if api_key:
                    self._client = openai.OpenAI(api_key=api_key)
            except Exception:
                pass
        return self._client

    # _clean_words: Removes punctuation and stop words from a string to prepare it for keyword matching.
    def _clean_words(self, text: str) -> set[str]:
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        stopwords = {"a", "an", "the", "and", "or", "but", "if", "then", "else", "is", "are", "was", "were", "be", "been", "being", "in", "on", "at", "to", "for", "of", "with", "about"}
        return {w for w in cleaned.split() if w not in stopwords and len(w) > 1}

    # search_memory
    # This method searches through the memories to find ones that are relevant to the query.
    # It uses OpenAI's embedding model to find the most similar memories.
    async def search(self, query: str, options: SearchOptions | None = None) -> list[MemoryEntry]:
        limit = 3 # Default to 3 search results if no options are provided.
        if options and "max_search_results" in options:
            limit = options["max_search_results"]

        client = self._get_client()
        if client: # If OpenAI is available -> Use semantic search
            try:
                # Lazily compute embeddings for any entries that are missing
                for entry in self._entries:
                    if entry not in self._embeddings:
                        # Compute embeddings
                        emb_resp = client.embeddings.create(
                            input=entry,
                            model="text-embedding-3-small"
                        )
                        self._embeddings[entry] = emb_resp.data[0].embedding

                # Compute query embedding
                q_resp = client.embeddings.create(
                    input=query,
                    model="text-embedding-3-small"
                )
                q_emb = q_resp.data[0].embedding

                # Calculate similarities
                scored_entries = []
                for entry in self._entries:
                    emb = self._embeddings.get(entry)
                    if emb:
                        sim = sum(x * y for x, y in zip(q_emb, emb))
                        scored_entries.append((entry, sim))

                scored_entries.sort(key=lambda x: x[1], reverse=True) # Return top results.
                return [MemoryEntry(content=entry) for entry, _ in scored_entries[:limit]]
            except Exception as e:
                # Fallback silently and use keyword search
                pass

        # Fallback: Use keyword search.
        query_words = self._clean_words(query)
        scored_fallback = []
        for entry in self._entries:
            entry_words = self._clean_words(entry)
            overlap = 0
            for qw in query_words:
                for ew in entry_words:
                    if qw == ew:
                        overlap += 2
                    elif len(qw) > 2 and len(ew) > 2 and (qw in ew or ew in qw):
                        overlap += 1
            if overlap > 0:
                scored_fallback.append((entry, overlap))

        scored_fallback.sort(key=lambda x: x[1], reverse=True)
        matches = [entry for entry, _ in scored_fallback]

        # Ultimate fallback: Simple substring match
        if not matches:
            matches = [entry for entry in self._entries if query.lower() in entry.lower()]

        return [MemoryEntry(content=content) for content in matches[:limit]]

    async def add(self, content: str, metadata: dict | None = None) -> None:
        self._entries.append(content)
        # Prefetch embedding for the new content asynchronously
        client = self._get_client()
        if client:
            try:
                emb_resp = client.embeddings.create(
                    input=content,
                    model="text-embedding-3-small"
                )
                self._embeddings[content] = emb_resp.data[0].embedding
            except Exception:
                pass

# Initialize the Store
def run_memory_demo():
    # 1. Initialize the custom store and add some default facts
    store = InMemoryStore()
    store._entries.append("User lives in Seattle.")
    store._entries.append("User has a dog named Buddy.")
    
    # 2. Setup the MemoryManager
    # The Memory Manager sits between the agent and the memory stores.
    memory_manager = MemoryManager(
        stores=[store],
        add_tool_config=True  # Enables the add_memory tool
    )
    
    api_key, model_name = get_openai_credentials()
    model = OpenAIModel(
        model_id=model_name,
        client_args={"api_key": api_key}
    )
    
    # Create the agent with the memory manager
    agent = Agent(
        model=model,
        system_prompt=(
            "You are a helpful assistant with access to the user's memories. "
            "Use search_memory to look up user facts, and add_memory to save new facts. "
            "Always search memory first when the user asks what you know about them or asks you to recall facts."
        ),
        memory_manager=memory_manager
    )
    
    print(f"Registered tools: {agent.tool_names}\n")
    
    # Run the queries
    print("=== Query 1: What does the agent remember? ===")
    query_1 = "Hi! Do you recall where I live and if I have any pets?"
    print(f"User: {query_1}")
    response_1 = agent(query_1)
    print(f"Agent: {response_1}\n")
    
    print("=== Query 2: Ask the agent to remember new facts ===")
    query_2 = "Please remember that my name is Gowtham and my favorite coding language is Python."
    print(f"User: {query_2}")
    response_2 = agent(query_2)
    print(f"Agent: {response_2}\n")
    
    print("=== Query 3: Ask the agent to recall the new facts ===")
    query_3 = "What is my name, and what is my favorite programming language?"
    print(f"User: {query_3}")
    response_3 = agent(query_3)
    print(f"Agent: {response_3}\n")
    
    print("=== Current Store Contents in memory ===")
    print(store._entries)

if __name__ == "__main__":
    run_memory_demo()
