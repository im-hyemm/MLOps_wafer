from contextlib import contextmanager
from typing import Dict, Optional

from config.db import DB_CONFIG, DB_POOL_MIN, DB_POOL_MAX, DB_CONNECT_TIMEOUT

try:
    from psycopg2 import pool as psycopg2_pool
except ImportError:  # pragma: no cover - library might be optional in some environments
    psycopg2_pool = None

_db_pool: Optional["psycopg2_pool.SimpleConnectionPool"] = None


def _get_connection_pool() -> Optional["psycopg2_pool.SimpleConnectionPool"]:
    global _db_pool

    if psycopg2_pool is None:
        return None

    if _db_pool is None:
        conn_kwargs = dict(DB_CONFIG)
        conn_kwargs.setdefault("connect_timeout", DB_CONNECT_TIMEOUT)

        try:
            _db_pool = psycopg2_pool.SimpleConnectionPool(
                DB_POOL_MIN,
                DB_POOL_MAX,
                **conn_kwargs,
            )
        except Exception as exc:  # pragma: no cover - initialization errors
            print(f"[DB] Failed to initialize connection pool: {exc}")
            _db_pool = None

    return _db_pool


@contextmanager
def get_connection():
    pool = _get_connection_pool()
    if pool is None:
        yield None
        return

    connection = pool.getconn()
    try:
        yield connection
    finally:
        pool.putconn(connection)


def fetch_lot_process_history(lot_code: str) -> Dict[str, Dict[str, str]]:
    """Return process history for a lot keyed by process step."""
    if not lot_code:
        return {}

    history: Dict[str, Dict[str, str]] = {}
    rows = []

    with get_connection() as connection:
        if connection is None:
            return history

        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT lp.step_name, lp.tool, lp.recipe
                    FROM lot l
                    JOIN lot_process_run lp ON lp.lot_id = l.lot_id
                    WHERE l.lot_code = %s
                    ORDER BY lp.run_id ASC
                    """,
                    (lot_code,),
                )
                rows = cursor.fetchall()
        except Exception as exc:
            print(f"[DB] Failed to fetch process history for lot {lot_code}: {exc}")
            return history

    for step_name, tool, recipe in rows:
        history[step_name] = {
            "tool": tool,
            "recipe": recipe,
        }

    return history
