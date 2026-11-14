"""Relay AI Universal Context Engine — Database Connection & Lifecycle Manager.

Provides thread-safe SQLite connection factory with WAL mode, foreign key cascades,
integrity verification, and corruption recovery.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Generator, List, Optional, Set, Union

from src.context.db.migrations import MigrationManager

logger = logging.getLogger(__name__)


def get_default_db_path() -> Path:
    """Resolve the default persistent context database path on the host system.

    On Windows, resolves to %LOCALAPPDATA%/RelayAI/relay_context.db.
    Fallback: ~/.relayai/relay_context.db.
    """
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        base_dir = Path(local_app_data) / "RelayAI"
    else:
        base_dir = Path.home() / ".relayai"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "relay_context.db"


def configure_connection_pragmas(conn: sqlite3.Connection, is_memory: bool = False) -> None:
    """Configure essential SQLite PRAGMA settings on a connection."""
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.execute("PRAGMA recursive_triggers = ON;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    if not is_memory:
        conn.execute("PRAGMA journal_mode = WAL;")


def check_database_integrity(conn: sqlite3.Connection) -> bool:
    """Execute SQLite PRAGMA quick_check to verify database file integrity.

    Returns:
        True if the database is sound, False if corruption is detected.
    """
    try:
        cursor = conn.execute("PRAGMA quick_check;")
        rows = cursor.fetchall()
        if not rows:
            return False
        first_res = rows[0][0]
        return str(first_res).strip().lower() == "ok"
    except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
        logger.error("Database integrity check failed with error: %s", e)
        return False


def get_db_connection(
    db_path: Optional[Union[str, Path]] = None,
    is_memory: Optional[bool] = None,
) -> sqlite3.Connection:
    """Create and configure an individual SQLite connection.

    Args:
        db_path: Path to database file or ':memory:'. If None, uses default path.
        is_memory: Explicit flag for in-memory mode. Auto-detected if omitted.

    Returns:
        Configured sqlite3.Connection instance.
    """
    if db_path is None:
        path_str = str(get_default_db_path())
        mem_flag = False
    elif db_path == ":memory:":
        path_str = ":memory:"
        mem_flag = True
    else:
        path_str = str(db_path)
        mem_flag = path_str == ":memory:" or "mode=memory" in path_str

    if is_memory is not None:
        mem_flag = is_memory

    if not mem_flag and path_str != ":memory:":
        parent = Path(path_str).parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path_str, timeout=10.0, check_same_thread=False)
    configure_connection_pragmas(conn, is_memory=mem_flag)
    return conn


class ContextDatabase:
    """Thread-safe SQLite database manager for Relay AI Universal Context Engine.

    Manages connection lifecycles across threads, enforces WAL concurrency,
    verifies file integrity on startup, recovers from corruption, and runs migrations.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize database manager.

        Args:
            db_path: Path to SQLite database file or ':memory:'.
                     Defaults to get_default_db_path().
        """
        self._raw_path = db_path
        self._lock = threading.RLock()
        self._local = threading.local()
        self._all_connections: Set[sqlite3.Connection] = set()
        self._initialized = False

        if db_path is None:
            self.db_path = get_default_db_path()
            self._is_memory = False
            self._connect_uri = False
            self._connect_path = str(self.db_path)
            self._keepalive_conn: Optional[sqlite3.Connection] = None
        elif str(db_path) == ":memory:":
            self.db_path = Path(":memory:")
            self._is_memory = True
            # Unique shared in-memory database per ContextDatabase instance
            mem_id = f"relay_mem_{id(self)}_{int(time.time() * 1000)}"
            self._connect_path = f"file:{mem_id}?mode=memory&cache=shared"
            self._connect_uri = True
            # Keep a persistent connection open so the shared in-memory DB is preserved
            self._keepalive_conn = sqlite3.connect(
                self._connect_path, uri=True, timeout=10.0, check_same_thread=False
            )
            configure_connection_pragmas(self._keepalive_conn, is_memory=True)
            self._all_connections.add(self._keepalive_conn)
        else:
            self.db_path = Path(db_path)
            self._is_memory = str(db_path) == ":memory:" or "mode=memory" in str(db_path)
            self._connect_uri = str(db_path).startswith("file:")
            self._connect_path = str(db_path)
            self._keepalive_conn = None

    def initialize(self) -> None:
        """Verify database integrity, recover from corruption if necessary, and apply migrations."""
        with self._lock:
            if not self._is_memory and self.db_path.exists():
                self._check_and_recover_file_integrity()

            # Ensure parent directory exists for file DBs
            if not self._is_memory and not self.db_path.parent.exists():
                self.db_path.parent.mkdir(parents=True, exist_ok=True)

            conn = self.get_connection()
            migration_manager = MigrationManager(conn)
            migration_manager.apply_pending_migrations()
            self._initialized = True

    def _check_and_recover_file_integrity(self) -> None:
        """Inspect file integrity; if corrupted, rename and recover."""
        if not self.db_path.exists() or self.db_path.stat().st_size == 0:
            return

        test_conn = None
        is_corrupt = False
        try:
            test_conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            if not check_database_integrity(test_conn):
                is_corrupt = True
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as err:
            logger.error("Database check failed with exception: %s", err)
            is_corrupt = True
        finally:
            if test_conn:
                test_conn.close()

        if is_corrupt:
            timestamp = int(time.time())
            corrupted_path = self.db_path.with_name(
                f"{self.db_path.stem}_corrupted_{timestamp}{self.db_path.suffix}"
            )
            logger.critical(
                "Database corruption detected at %s. Renaming corrupted file to %s and initializing clean database.",
                self.db_path,
                corrupted_path,
            )
            # Rename corrupted file and associated WAL/SHM files
            try:
                os.rename(str(self.db_path), str(corrupted_path))
                wal_file = Path(f"{self.db_path}-wal")
                if wal_file.exists():
                    os.rename(str(wal_file), str(corrupted_path.with_name(f"{corrupted_path.name}-wal")))
                shm_file = Path(f"{self.db_path}-shm")
                if shm_file.exists():
                    os.rename(str(shm_file), str(corrupted_path.with_name(f"{corrupted_path.name}-shm")))
            except OSError as os_err:
                logger.error("Failed to rename corrupted database files: %s", os_err)

    def get_connection(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection, creating and configuring it if necessary."""
        conn = getattr(self._local, "connection", None)
        if conn is None:
            if self._connect_uri:
                conn = sqlite3.connect(
                    self._connect_path, uri=True, timeout=10.0, check_same_thread=False
                )
            else:
                conn = sqlite3.connect(
                    self._connect_path, timeout=10.0, check_same_thread=False
                )

            configure_connection_pragmas(conn, is_memory=self._is_memory)
            self._local.connection = conn
            with self._lock:
                self._all_connections.add(conn)

        return conn

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing an active connection wrapped in an atomic transaction."""
        if not self._initialized:
            self.initialize()

        conn = self.get_connection()
        try:
            with conn:
                yield conn
        except Exception:
            conn.rollback()
            raise

    def execute(self, sql: str, params: Union[tuple, dict] = ()) -> sqlite3.Cursor:
        """Execute a single SQL statement inside a transaction."""
        with self.connection() as conn:
            return conn.execute(sql, params)

    def executemany(self, sql: str, seq_of_params: Any) -> sqlite3.Cursor:
        """Execute a parameterized SQL statement across a sequence of parameters."""
        with self.connection() as conn:
            return conn.executemany(sql, seq_of_params)

    def fetchall(self, sql: str, params: Union[tuple, dict] = ()) -> List[sqlite3.Row]:
        """Execute a query and fetch all result rows."""
        conn = self.get_connection()
        if not self._initialized:
            self.initialize()
        cursor = conn.execute(sql, params)
        return cursor.fetchall()

    def fetchone(self, sql: str, params: Union[tuple, dict] = ()) -> Optional[sqlite3.Row]:
        """Execute a query and fetch the first result row."""
        conn = self.get_connection()
        if not self._initialized:
            self.initialize()
        cursor = conn.execute(sql, params)
        return cursor.fetchone()

    def close(self) -> None:
        """Close all connections opened by this manager."""
        with self._lock:
            for conn in list(self._all_connections):
                try:
                    conn.close()
                except Exception as e:
                    logger.debug("Error closing database connection: %s", e)
            self._all_connections.clear()
            self._keepalive_conn = None
            if hasattr(self._local, "connection"):
                delattr(self._local, "connection")
            self._initialized = False


def initialize_database(db_path: Optional[Union[str, Path]] = None) -> ContextDatabase:
    """Create, initialize, and return a ContextDatabase instance."""
    db = ContextDatabase(db_path)
    db.initialize()
    return db
