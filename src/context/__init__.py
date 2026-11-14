"""Relay AI Universal Context Engine (UCE).

Provides persistent memory, multi-project context isolation, document intelligence,
FTS5 BM25 retrieval, and multi-turn conversation persistence.
"""

from src.context.db.connection import (
    ContextDatabase,
    check_database_integrity,
    get_db_connection,
    get_default_db_path,
    initialize_database,
)
from src.context.db.migrations import MigrationManager
from src.context.db.schema import SCHEMA_VERSION

__all__ = [
    "ContextDatabase",
    "MigrationManager",
    "SCHEMA_VERSION",
    "check_database_integrity",
    "get_db_connection",
    "get_default_db_path",
    "initialize_database",
]
