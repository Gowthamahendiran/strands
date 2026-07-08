import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any
import boto3
from botocore.exceptions import ClientError

from strands.types.exceptions import SessionException
from strands.types.session import Session, SessionAgent, SessionMessage
from strands.session.repository_session_manager import RepositorySessionManager
from strands.session.session_repository import SessionRepository

if TYPE_CHECKING:
    from strands.multiagent import MultiAgentBase

logger = logging.getLogger(__name__)

# SessionException: It is a built-in Strands exception class for handling session-related errors.
# Session: It is a built-in Strands data model for representing a session.
# SessionAgent: It is a built-in Strands data model for representing an agent within a session.
# SessionMessage: It is a built-in Strands data model for representing a message within a session.
# SessionRepository: It is a built-in Strands interface that defines the basic operations required for a session manager.
# RepositorySessionManager: It is a built-in Strands helper class that manages agent sessions and tells the database where and when to save or load chat data.

class DynamoDBSessionManager(RepositorySessionManager, SessionRepository):
    """DynamoDB-based session manager storing the entire session state inside a single record."""

    def __init__(
        self,
        session_id: str,
        table_name: str = "strandssession",
        region_name: str = None,
        client_id: str = "default_client",
        **kwargs: Any,
    ):
        """Initialize the DynamoDBSessionManager.

        Args:
            session_id: ID for the session (Partition Key).
            table_name: The DynamoDB table name. Defaults to "strandssession".
            region_name: The AWS region.
            client_id: The client identifier (Sort Key).
            **kwargs: Additional keyword arguments.
        """
        self.table_name = table_name
        self.client_id = client_id
        self.dynamodb = boto3.resource("dynamodb", region_name=region_name)
        self.table = self.dynamodb.Table(table_name)
        
        super().__init__(session_id=session_id, session_repository=self, **kwargs)

    def create_session(self, session: Session, **kwargs: Any) -> Session:
        """Create a new session record in DynamoDB."""
        now = datetime.utcnow().isoformat()
        try:
            self.table.put_item(
                Item={
                    "SessionId": session.session_id,
                    "ClientId": self.client_id,
                    "CreatedAt": now,
                    "LastUpdated": now,
                    "MessageCount": 0,
                    "History": [],
                    "AgentState": {},
                    "MultiAgentState": {},
                    "strands_session_meta": json.dumps(session.to_dict(), ensure_ascii=False)
                },
                ConditionExpression="attribute_not_exists(SessionId) AND attribute_not_exists(ClientId)"
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise SessionException(f"Session {session.session_id} with Client ID {self.client_id} already exists")
            raise SessionException(f"Failed to create session in DynamoDB: {str(e)}") from e
            
        return session

    def read_session(self, session_id: str, **kwargs: Any) -> Session | None:
        """Read session metadata from the single DynamoDB record."""
        try:
            response = self.table.get_item(
                Key={
                    "SessionId": session_id,
                    "ClientId": self.client_id
                }
            )
            item = response.get("Item")
            if not item:
                return None
            meta_str = item.get("strands_session_meta")
            if meta_str:
                session_data = json.loads(meta_str)
                return Session.from_dict(session_data)
            return Session(session_id=session_id)
        except ClientError as e:
            logger.error(f"Error reading session {session_id} with Client ID {self.client_id} from DynamoDB: {e}")
            return None

    def delete_session(self, session_id: str, **kwargs: Any) -> None:
        """Delete the single session record from DynamoDB."""
        try:
            self.table.delete_item(
                Key={
                    "SessionId": session_id,
                    "ClientId": self.client_id
                }
            )
        except ClientError as e:
            raise SessionException(f"Failed to delete session {session_id} with Client ID {self.client_id} from DynamoDB: {str(e)}") from e

    def create_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        """Create or initialize agent metadata inside the session record."""
        agent_id = session_agent.agent_id
        try:
            self.table.update_item(
                Key={
                    "SessionId": session_id,
                    "ClientId": self.client_id
                },
                UpdateExpression="SET AgentState.#agent_id = :agent_data",
                ExpressionAttributeNames={"#agent_id": agent_id},
                ExpressionAttributeValues={":agent_data": json.dumps(session_agent.to_dict(), ensure_ascii=False)}
            )
        except ClientError as e:
            raise SessionException(f"Failed to create agent {agent_id} in DynamoDB: {str(e)}") from e

    def read_agent(self, session_id: str, agent_id: str, **kwargs: Any) -> SessionAgent | None:
        """Read agent metadata from the session record."""
        try:
            response = self.table.get_item(
                Key={
                    "SessionId": session_id,
                    "ClientId": self.client_id
                }
            )
            item = response.get("Item")
            if not item:
                return None
            agent_states = item.get("AgentState", {})
            agent_str = agent_states.get(agent_id)
            if not agent_str:
                return None
            return SessionAgent.from_dict(json.loads(agent_str))
        except ClientError as e:
            logger.error(f"Error reading agent {agent_id} in session {session_id}: {e}")
            return None

    def update_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        """Update agent metadata inside the session record."""
        agent_id = session_agent.agent_id
        previous_agent = self.read_agent(session_id=session_id, agent_id=agent_id)
        if previous_agent is None:
            raise SessionException(f"Agent {agent_id} in session {session_id} does not exist")

        session_agent.created_at = previous_agent.created_at
        try:
            self.table.update_item(
                Key={
                    "SessionId": session_id,
                    "ClientId": self.client_id
                },
                UpdateExpression="SET AgentState.#agent_id = :agent_data",
                ExpressionAttributeNames={"#agent_id": agent_id},
                ExpressionAttributeValues={":agent_data": json.dumps(session_agent.to_dict(), ensure_ascii=False)}
            )
        except ClientError as e:
            raise SessionException(f"Failed to update agent {agent_id} in DynamoDB: {str(e)}") from e

    def create_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any) -> None:
        """Append a new message to the History list in the DynamoDB record."""
        msg_dict = session_message.to_dict()
        
        role = msg_dict.get("message", {}).get("role", "")
        msg_type = "human" if role == "user" else "ai"

        # Content extraction
        content_list = msg_dict.get("message", {}).get("content", [])
        content_str = ""
        if content_list and isinstance(content_list, list):
            content_str = content_list[0].get("text", "")
        elif isinstance(content_list, str):
            content_str = content_list

        if not content_str or not content_str.strip():
            logger.debug("Skipping message with empty content (likely tool call/result)")
            return

        # Extraction of usage and metrics metadata
        metadata = msg_dict.get("message", {}).get("metadata", {})
        usage = metadata.get("usage", {})
        metrics = metadata.get("metrics", {})

        # Langchain-compatible structure (simplified, without id and name fields)
        data_map = {
            "additional_kwargs": {},
            "content": content_str,
            "example": False,
            "response_metadata": {
                "usage": usage,
                "metrics": metrics
            },
            "type": msg_type
        }

        history_item = {
            "type": msg_type,
            "data": data_map,
            "message_id": session_message.message_id,
            "redact_message": msg_dict.get("redact_message"),
            "created_at": msg_dict.get("created_at"),
            "updated_at": msg_dict.get("updated_at")
        }

        now = datetime.utcnow().isoformat()
        try:
            self.table.update_item(
                Key={
                    "SessionId": session_id,
                    "ClientId": self.client_id
                },
                UpdateExpression="SET History = list_append(if_not_exists(History, :empty_list), :new_msg), LastUpdated = :now, MessageCount = MessageCount + :one",
                ExpressionAttributeValues={
                    ":new_msg": [history_item],
                    ":empty_list": [],
                    ":now": now,
                    ":one": 1
                }
            )
        except ClientError as e:
            raise SessionException(f"Failed to append message in DynamoDB: {str(e)}") from e

    def read_message(self, session_id: str, agent_id: str, message_id: int, **kwargs: Any) -> SessionMessage | None:
        """Find and read a message from the History list by its ID."""
        try:
            response = self.table.get_item(
                Key={
                    "SessionId": session_id,
                    "ClientId": self.client_id
                }
            )
            item = response.get("Item")
            if not item:
                return None
            history = item.get("History", [])
            for h_item in history:
                if h_item.get("message_id") == message_id:
                    return self._reconstruct_message(h_item)
            return None
        except ClientError as e:
            logger.error(f"Error reading message {message_id} from DynamoDB: {e}")
            return None

    def update_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any) -> None:
        """Update a specific message in the History list."""
        message_id = session_message.message_id
        try:
            response = self.table.get_item(
                Key={
                    "SessionId": session_id,
                    "ClientId": self.client_id
                }
            )
            item = response.get("Item")
            if not item:
                raise SessionException("Session does not exist")
            
            history = item.get("History", [])
            updated = False
            for idx, h_item in enumerate(history):
                if h_item.get("message_id") == message_id:
                    msg_dict = session_message.to_dict()
                    role = msg_dict.get("message", {}).get("role", "")
                    msg_type = "human" if role == "user" else "ai"
                    
                    content_list = msg_dict.get("message", {}).get("content", [])
                    content_str = ""
                    if content_list and isinstance(content_list, list):
                        content_str = content_list[0].get("text", "")
                    elif isinstance(content_list, str):
                        content_str = content_list

                    metadata = msg_dict.get("message", {}).get("metadata", {})
                    usage = metadata.get("usage", {})
                    metrics = metadata.get("metrics", {})

                    h_item["type"] = msg_type
                    h_item["data"] = {
                        "additional_kwargs": {},
                        "content": content_str,
                        "example": False,
                        "response_metadata": {
                            "usage": usage,
                            "metrics": metrics
                        },
                        "type": msg_type
                    }
                    h_item["redact_message"] = msg_dict.get("redact_message")
                    h_item["updated_at"] = msg_dict.get("updated_at")
                    updated = True
                    break
                    
            if not updated:
                raise SessionException(f"Message {message_id} does not exist in history")
                
            now = datetime.utcnow().isoformat()
            self.table.update_item(
                Key={
                    "SessionId": session_id,
                    "ClientId": self.client_id
                },
                UpdateExpression="SET History = :history, LastUpdated = :now",
                ExpressionAttributeValues={
                    ":history": history,
                    ":now": now
                }
            )
        except ClientError as e:
            raise SessionException(f"Failed to update message in DynamoDB: {str(e)}") from e

    def _reconstruct_message(self, h_item: dict) -> SessionMessage:
        """Reconstruct a SessionMessage from the simplified history format."""
        msg_type = h_item.get("type", "")
        data = h_item.get("data", {})
        role = "user" if msg_type == "human" else "assistant"
        
        response_metadata = data.get("response_metadata", {})
        usage = response_metadata.get("usage", {})
        metrics = response_metadata.get("metrics", {})

        # Build the Standard Dict
        msg_dict = {
            "message": {
                "role": role,
                "content": [{"text": data.get("content", "")}],
                "metadata": {
                    "usage": usage,
                    "metrics": metrics
                }
            },
            "message_id": h_item.get("message_id", 0),
            "redact_message": None,
            "created_at": h_item.get("created_at") or datetime.utcnow().isoformat(),
            "updated_at": h_item.get("updated_at") or datetime.utcnow().isoformat()
        }
        # Native Strands object.
        return SessionMessage.from_dict(msg_dict)

    def list_messages(
        self, session_id: str, agent_id: str, limit: int | None = None, offset: int = 0, **kwargs: Any
    ) -> list[SessionMessage]:
        """List messages for an agent with pagination."""
        try:
            response = self.table.get_item(
                Key={
                    "SessionId": session_id,
                    "ClientId": self.client_id
                }
            )
            item = response.get("Item")
            if not item:
                return []
            history = item.get("History", [])
            
            # Sort history by message_id to ensure order
            history.sort(key=lambda x: x.get("message_id", 0))
            
            # Filter and paginate
            if limit is not None:
                history = history[offset : offset + limit]
            else:
                history = history[offset:]
                
            messages = []
            for h_item in history:
                messages.append(self._reconstruct_message(h_item))
            return messages
        except ClientError as e:
            raise SessionException(f"Failed to list messages from DynamoDB: {str(e)}") from e

    def create_multi_agent(self, session_id: str, multi_agent: "MultiAgentBase", **kwargs: Any) -> None:
        """Store multi-agent state inside the session record."""
        multi_agent_id = multi_agent.id
        try:
            self.table.update_item(
                Key={
                    "SessionId": session_id,
                    "ClientId": self.client_id
                },
                UpdateExpression="SET MultiAgentState.#id = :state",
                ExpressionAttributeNames={"#id": multi_agent_id},
                ExpressionAttributeValues={":state": json.dumps(multi_agent.serialize_state(), ensure_ascii=False)}
            )
        except ClientError as e:
            raise SessionException(f"Failed to create multi-agent state in DynamoDB: {str(e)}") from e

    def read_multi_agent(self, session_id: str, multi_agent_id: str, **kwargs: Any) -> dict[str, Any] | None:
        """Read multi-agent state from the session record."""
        try:
            response = self.table.get_item(
                Key={
                    "SessionId": session_id,
                    "ClientId": self.client_id
                }
            )
            item = response.get("Item")
            if not item:
                return None
            multi_states = item.get("MultiAgentState", {})
            state_str = multi_states.get(multi_agent_id)
            if not state_str:
                return None
            return json.loads(state_str)
        except ClientError as e:
            logger.error(f"Error reading multi-agent state {multi_agent_id}: {e}")
            return None

    def update_multi_agent(self, session_id: str, multi_agent: "MultiAgentBase", **kwargs: Any) -> None:
        """Update multi-agent state inside the session record."""
        multi_agent_state = multi_agent.serialize_state()
        previous_state = self.read_multi_agent(session_id=session_id, multi_agent_id=multi_agent.id)
        if previous_state is None:
            raise SessionException(f"MultiAgent state {multi_agent.id} in session {session_id} does not exist")
            
        try:
            self.table.update_item(
                Key={
                    "SessionId": session_id,
                    "ClientId": self.client_id
                },
                UpdateExpression="SET MultiAgentState.#id = :state",
                ExpressionAttributeNames={"#id": multi_agent.id},
                ExpressionAttributeValues={":state": json.dumps(multi_agent_state, ensure_ascii=False)}
            )
        except ClientError as e:
            raise SessionException(f"Failed to update multi-agent state in DynamoDB: {str(e)}") from e
