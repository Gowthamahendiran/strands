import boto3
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def create_session_table(table_name: str = "strandssession", region_name: str = None) -> None:
    """Create the DynamoDB session table if it doesn't already exist.
    
    If the table exists but has the wrong key schema,
    it will be deleted and recreated.
    """
    dynamodb = boto3.resource("dynamodb", region_name=region_name)
    
    try:
        table = dynamodb.Table(table_name)
        table.load()
        
        # Verify the key schema (Partition Key: SessionId, Sort Key: ClientId)
        key_names = [k["AttributeName"] for k in table.key_schema]
        if key_names != ["SessionId", "ClientId"]:
            print(f"Table '{table_name}' exists but has incorrect key schema {key_names}. Re-creating with HASH 'SessionId' and RANGE 'ClientId'...")
            table.delete()
            table.wait_until_not_exists()
        else:
            print(f"Table '{table_name}' already exists with correct schema ('SessionId' HASH, 'ClientId' RANGE).")
            return
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise e
            
    print(f"Creating DynamoDB table '{table_name}' with HASH key 'SessionId' and RANGE key 'ClientId'...")
    
    # Create the table with SessionId as Partition Key, ClientId as Sort Key, and PAY_PER_REQUEST billing
    table = dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "SessionId", "KeyType": "HASH"},
            {"AttributeName": "ClientId", "KeyType": "RANGE"}
        ],
        AttributeDefinitions=[
            {"AttributeName": "SessionId", "AttributeType": "S"},
            {"AttributeName": "ClientId", "AttributeType": "S"}
        ],
        BillingMode="PAY_PER_REQUEST"
    )
    
    # Wait for creation to complete
    print(f"Waiting for table '{table_name}' to be active...")
    table.wait_until_exists()
    print(f"Table '{table_name}' is now active and ready.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_session_table()
