import boto3

def create_memory_table(table_name: str = "strandsmemory", region_name: str = None) -> None:
    """Create the DynamoDB memory table if it doesn't already exist."""
    dynamodb = boto3.resource("dynamodb", region_name=region_name)
    try:
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "ClientId", "KeyType": "HASH"},
                {"AttributeName": "MemoryId", "KeyType": "RANGE"}
            ],
            AttributeDefinitions=[
                {"AttributeName": "ClientId", "AttributeType": "S"},
                {"AttributeName": "MemoryId", "AttributeType": "S"}
            ],
            BillingMode="PAY_PER_REQUEST"
        )
        table.wait_until_exists()
        print(f"Created DynamoDB Memory Table '{table_name}' successfully.")
    except dynamodb.meta.client.exceptions.ResourceInUseException:
        print(f"DynamoDB Memory Table '{table_name}' already exists.")
