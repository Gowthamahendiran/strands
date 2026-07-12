import json
import logging
import os
import sys
from functools import lru_cache
from typing import Any

import boto3

# Add backend directory to sys.path to allow imports from app.config
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from dotenv import load_dotenv
dotenv_path = os.path.join(backend_dir, ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# pyrefly: ignore [missing-import]
from app.config import first_present, get_settings

logger = logging.getLogger(__name__)

os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


class SecretLoadError(RuntimeError):
    """Raised when the configured AWS secret cannot be loaded."""


def get_secret(secret_name: str, region_name: str = "us-east-1") -> dict[str, Any]:
    try:
        client = boto3.client("secretsmanager", region_name=region_name)
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        secret_string = get_secret_value_response.get("SecretString")
        if secret_string:
            parsed = json.loads(secret_string)
            if not isinstance(parsed, dict):
                raise ValueError("Secret string must be a JSON object")
            return parsed
        raise ValueError("Secret string is empty")
    except Exception as exc:
        logger.error("Failed to retrieve secret: %s", exc)
        raise


@lru_cache
def get_llm_secret() -> dict[str, Any]:
    settings = get_settings()
    try:
        return get_secret(settings.aws_secret_name, region_name=settings.aws_region)
    except Exception as exc:
        raise SecretLoadError(f"Unable to load AWS secret '{settings.aws_secret_name}'") from exc


def get_openai_credentials() -> tuple[str, str]:
    settings = get_settings()
    secret = get_llm_secret()

    api_key = first_present(secret, "OPENAI_API_KEY", "openai_api_key", "LLM_API_KEY", "api_key")
    model = first_present(secret, "OPENAI_MODEL", "openai_model", "LLM_MODEL", "model")

    api_key = api_key or settings.openai_api_key
    model = model or settings.openai_model

    if not api_key:
        raise SecretLoadError(
            "No LLM API key found. Add OPENAI_API_KEY to the AIT-RAG-AGENT-CORE secret, "
            "or set OPENAI_API_KEY locally for development."
        )

    os.environ["OPENAI_API_KEY"] = str(api_key)
    return str(api_key), str(model)
