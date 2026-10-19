from pydantic_core import Url
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, field_validator
import os

class Env(BaseModel):
    openai_api_key : Url | None = os.getenv("OPENAI_API_KEY")
    gemini_api_key : Url | None = os.getenv("GEMINI_API_KEY")
    anthropic_api_key : Url | None = os.getenv("ANTHROPIC_API_KEY")
    groq_api_key : Url | None = os.getenv("GROQ_API_KEY")
    openrouter_api_key : Url | None = os.getenv("OPENROUTER_API_KEY")
    anthropic_api_key: Url | None = os.getenv("ANTHROPIC_AGENTIC_SDK")

    @field_validator("openai_api_key", "gemini_api_key", "anthropic_api_key", "groq_api_key", "openrouter_api_key")
    @classmethod
    def validate_optional(cls, v: Url | None) -> Url | None:
        if v is None:
            return None

        return v

    @field_validator("openai_api_key", "gemini_api_key", "anthropic_api_key", "groq_api_key", "openrouter_api_key")
    @classmethod
    def validate_strict(cls, v: Url) -> Url:
        return v


env = Env()
