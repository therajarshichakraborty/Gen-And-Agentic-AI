import asyncio
import importlib.util
from pathlib import Path

_env_path = Path(__file__).resolve().parents[2] / "env.config.py"
_spec = importlib.util.spec_from_file_location("env_config", _env_path)
_env_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_env_module)
env = _env_module.env

from claude_agent_sdk import query , ClaudeAgentOptions, AssistantMessage , ResultMessage

async def init_agent():
    async for message in query(
        prompt="Review utils.py for bugs that would cause crashes. Fix any issues you find.",
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Edit", "Glob"],
            permission_mode="acceptEdits"
        )
    ):
        
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text)
                elif hasattr(block, "name"):
                    print(f"Tool: {block.name}")
                    
        elif isinstance(message, ResultMessage):
            print(f"Done: {message.subtype}")
            

asyncio.run(init_agent())