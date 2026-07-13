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

    async def _check_duplicate_llm(self, existing: str, new: str) -> bool:
        """Lightweight LLM call to verify if the new fact is a duplicate of the existing fact."""
        try:
            try:
                from strands_secrets import get_openai_credentials
                _, model_name = get_openai_credentials()
            except Exception:
                model_name = "gpt-4o-mini"
            
            prompt = (
                "Compare the following two facts about a user and determine if the 'New Fact' is a duplicate of the 'Existing Fact'.\n"
                "Respond with 'DUPLICATE' if the new fact conveys the same information or is a subset of the existing fact.\n"
                "Respond with 'NEW' if the new fact contains new, different, or contradictory information.\n\n"
                f"Existing Fact: {existing}\n"
                f"New Fact: {new}\n\n"
                "Response (DUPLICATE or NEW):"
            )
            resp = self.openai_client.chat.completions.create(
                model=model_name or "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0
            )
            result = resp.choices[0].message.content.strip().upper()
            return "DUPLICATE" in result
        except Exception as e:
            print(f"Error in duplicate LLM check: {e}")
            return False

    async def add(self, content: str, metadata: dict | None = None) -> None:
        """Embed the memory text content, check for duplicates, and upsert it into the Pinecone index."""
        try:
            # 1. Compute embedding vector
            emb_resp = self.openai_client.embeddings.create(
                input=content,
                model="text-embedding-3-small",
                dimensions=512
            )
            embedding = emb_resp.data[0].embedding

            # 2. Query Pinecone for existing similar memories to check for duplicates
            response = self.index.query(
                vector=embedding,
                top_k=5,
                filter={"client_id": {"$eq": self.client_id}},
                include_metadata=True
            )
            
            is_duplicate = False
            matches = response.get("matches", [])
            for match in matches:
                match_content = match.get("metadata", {}).get("content", "")
                score = match.get("score", 0.0)
                if not match_content:
                    continue
                
                # Check exact case-insensitive match
                if content.strip().lower() == match_content.strip().lower():
                    is_duplicate = True
                    break
                
                # Compute cosine similarity (dot product of normalized vectors)
                if score >= 0.95:
                    is_duplicate = True
                    break
                elif score >= 0.65:
                    # Let LLM verify
                    if await self._check_duplicate_llm(match_content, content):
                        is_duplicate = True
                        break

            if is_duplicate:
                return

            # 3. Generate unique memory ID
            memory_id = f"mem_{uuid.uuid4().hex}"

            # 4. Store vector with metadata in Pinecone
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



