import asyncio
from backend.mcp_client import get_all_tools


if __name__ == "__main__":
    asyncio.run(get_all_tools())