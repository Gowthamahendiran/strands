import uuid
import boto3
import openai
from datetime import datetime
from decimal import Decimal
from typing import Any
from strands.memory.types import MemoryEntry, SearchOptions

class DynamoDBMemoryStore:
    """A persistent MemoryStore implementation storing memories in a dedicated DynamoDB table."""
    name = "preferences"
    description = "User preferences, personal information, and stable facts."
    max_search_results = 3
    writable = True
    extraction = None

    def __init__(self, table_name: str, client_id: str, api_key: str, region_name: str = None):
        self.table_name = table_name
        self.client_id = client_id
        self.dynamodb = boto3.resource("dynamodb", region_name=region_name)
        self.table = self.dynamodb.Table(table_name)
        self.openai_client = openai.OpenAI(api_key=api_key)

    async def search(self, query: str, options: SearchOptions | None = None) -> list[MemoryEntry]:
        """Query all user memories, calculate similarity, and return matches."""
        limit = self.max_search_results
        if options and "max_search_results" in options:
            limit = options["max_search_results"]

        try:
            # 1. Fetch all memory entries for this user/client from DynamoDB
            response = self.table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key("ClientId").eq(self.client_id)
            )
            items = response.get("Items", [])
            if not items:
                return []

            # 2. Get vector embedding for the query
            q_resp = self.openai_client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
            q_emb = q_resp.data[0].embedding

            # 3. Calculate cosine similarities (dot product) in memory
            scored_entries = []
            for item in items:
                db_emb = item.get("embedding")
                if db_emb:
                    # Convert Decimals back to floats for calculation
                    emb = [float(val) for val in db_emb]
                    # Dot product (both vectors are normalized by OpenAI, so similarity = dot product)
                    sim = sum(x * y for x, y in zip(q_emb, emb))
                    scored_entries.append((item.get("content"), sim))

            # 4. Sort and return top candidates
            scored_entries.sort(key=lambda x: x[1], reverse=True)
            return [MemoryEntry(content=content) for content, _ in scored_entries[:limit]]
        except Exception as e:
            print(f"Error in DynamoDBMemoryStore search: {e}")
            return []

    async def add(self, content: str, metadata: dict | None = None) -> None:
        """Embed the memory text content and put it as a new row in DynamoDB memory table."""
        try:
            # 1. Compute embedding vector
            emb_resp = self.openai_client.embeddings.create(
                input=content,
                model="text-embedding-3-small"
            )
            embedding = emb_resp.data[0].embedding
            
            # Convert float lists to Decimal lists to comply with DynamoDB / boto3 numeric types
            embedding_decimals = [Decimal(str(val)) for val in embedding]

            # 2. Generate unique memory ID
            memory_id = f"mem_{uuid.uuid4().hex}"

            # 3. Store item in strandsmemory table
            self.table.put_item(
                Item={
                    "ClientId": self.client_id,
                    "MemoryId": memory_id,
                    "content": content,
                    "embedding": embedding_decimals,
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            print(f"[Memory Store] Saved new memory: '{content}' (ID: {memory_id})")
        except Exception as e:
            print(f"Error in DynamoDBMemoryStore add: {e}")
