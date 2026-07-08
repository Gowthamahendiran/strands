import uuid
from datetime import datetime
import openai
from pinecone import Pinecone
from strands.memory.types import MemoryEntry, SearchOptions

class PineconeMemoryStore:
    """A persistent MemoryStore implementation storing memories in Pinecone."""
    name = "preferences"
    description = "User preferences, personal information, and stable facts."
    max_search_results = 3
    writable = True
    extraction = None

    def __init__(self, index_name: str, client_id: str, api_key: str, openai_api_key: str):
        self.index_name = index_name
        self.client_id = client_id
        
        # Initialize Pinecone Client
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        
        # Initialize OpenAI Client
        self.openai_client = openai.OpenAI(api_key=openai_api_key)

    async def search(self, query: str, options: SearchOptions | None = None) -> list[MemoryEntry]:
        """Search Pinecone for memories matching the query, filtered by user/client_id."""
        limit = self.max_search_results
        if options and "max_search_results" in options:
            limit = options["max_search_results"]

        try:
            # 1. Get vector embedding for the query
            q_resp = self.openai_client.embeddings.create(
                input=query,
                model="text-embedding-3-small",
                dimensions=512
            )
            q_emb = q_resp.data[0].embedding

            # 2. Query Pinecone index using metadata filter for user ID
            response = self.index.query(
                vector=q_emb,
                top_k=limit,
                filter={"client_id": {"$eq": self.client_id}},
                include_metadata=True
            )

            # 3. Format and return results
            results = []
            for match in response.get("matches", []):
                metadata = match.get("metadata", {})
                content = metadata.get("content")
                if content:
                    results.append(MemoryEntry(content=content))
            return results
        except Exception as e:
            print(f"Error in PineconeMemoryStore search: {e}")
            return []

    async def add(self, content: str, metadata: dict | None = None) -> None:
        """Embed the memory text content and upsert it into the Pinecone index."""
        try:
            # 1. Compute embedding vector
            emb_resp = self.openai_client.embeddings.create(
                input=content,
                model="text-embedding-3-small",
                dimensions=512
            )
            embedding = emb_resp.data[0].embedding

            # 2. Generate unique memory ID
            memory_id = f"mem_{uuid.uuid4().hex}"

            # 3. Store vector with metadata in Pinecone
            self.index.upsert(
                vectors=[
                    {
                        "id": memory_id,
                        "values": embedding,
                        "metadata": {
                            "content": content,
                            "client_id": self.client_id,
                            "created_at": datetime.utcnow().isoformat()
                        }
                    }
                ]
            )
            print(f"[Pinecone Memory Store] Saved new memory: '{content}' (ID: {memory_id})")
        except Exception as e:
            print(f"Error in PineconeMemoryStore add: {e}")
