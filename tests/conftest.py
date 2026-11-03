import os

import psycopg_pool
from langgraph.checkpoint.postgres import PostgresSaver


os.environ.setdefault(
    "DATABASE_URL",
    "postgresql://test:test@localhost:5432/test",
)
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("JWT_SECRET_KEY", "test-only-secret-key-at-least-32-bytes")


class _PoolStub:
    check_connection = staticmethod(lambda connection: None)

    def __init__(self, *args, **kwargs):
        pass

    def connection(self):
        raise RuntimeError("The test suite does not use a database connection.")

    def close(self):
        pass


psycopg_pool.ConnectionPool = _PoolStub
PostgresSaver.setup = lambda self: None
