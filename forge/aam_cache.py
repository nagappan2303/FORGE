"""
Disk-backed AAM result cache.

Why
---
Inline AAM is the dominant cost of seeded iteration. Many
enumerated reactions across iters and across runs are canonically
identical: same sorted reactant SMILES, same sorted product SMILES.
Caching results keyed on the canonical reaction string lets us skip
the mapper for repeats.

Storage
-------
A single sqlite file per run (default: <out_dir>/aam_cache.sqlite),
WAL-mode for concurrent multi-process access. Each forked worker opens
its own connection. Reads are lock-free under WAL; writes serialize via
SQLite's built-in busy-timeout.

Schema
------
    aam_cache(
        rxn_key   TEXT PRIMARY KEY,    -- canonical reactant.smi.>>product.smi
        allowed   INTEGER NOT NULL,    -- 0/1
        status    TEXT,                -- 'allowed'|'rejected'|'timeout'|'error'|...
        elapsed   REAL,
        ts        REAL                 -- unix epoch
    )
"""
from __future__ import annotations

import os
import sqlite3
import time
from typing import Optional, Tuple


_CREATE = """
CREATE TABLE IF NOT EXISTS aam_cache (
    rxn_key TEXT PRIMARY KEY,
    allowed INTEGER NOT NULL,
    status  TEXT,
    elapsed REAL,
    ts      REAL
)
"""

# per-process connection (forked workers each open their own)
_CONN: Optional[sqlite3.Connection] = None
_PATH: Optional[str] = None


# Bump whenever the mapper's verdict logic changes: cached verdicts computed
# by a different mapper version are then invalidated automatically instead of
# being replayed against the new logic.
MAPPER_VERSION = "1.0.0"


def _connect(path: str) -> sqlite3.Connection:
    """Open (or reopen) the cache for the current process. WAL + busy timeout."""
    global _CONN, _PATH
    if _CONN is not None and _PATH == path:
        return _CONN
    if _CONN is not None:
        try: _CONN.close()
        except Exception: pass
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    con = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    # WAL: many readers + one writer at a time, no global lock
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA busy_timeout=30000")
    except Exception:
        pass
    con.execute(_CREATE)
    # version gate: drop stale verdicts computed by a different mapper version
    con.execute("CREATE TABLE IF NOT EXISTS cache_meta (key TEXT PRIMARY KEY, value TEXT)")
    row = con.execute("SELECT value FROM cache_meta WHERE key='mapper_version'").fetchone()
    if row is None or row[0] != MAPPER_VERSION:
        # A missing version row on a NON-EMPTY cache is treated exactly
        # like an explicit version mismatch: its verdicts are dropped. A
        # truly fresh cache has zero rows, so the wipe is a no-op there.
        n_old = con.execute("SELECT count(*) FROM aam_cache").fetchone()[0]
        if n_old:
            old = row[0] if row is not None else "<unversioned>"
            print(f"[aam_cache] mapper version changed ({old} -> {MAPPER_VERSION}); "
                  f"dropping {n_old} stale cached verdicts in {path}", flush=True)
            con.execute("DELETE FROM aam_cache")
        con.execute("INSERT OR REPLACE INTO cache_meta(key, value) VALUES ('mapper_version', ?)",
                    (MAPPER_VERSION,))
    _CONN = con
    _PATH = path
    return con


def init(path: str) -> None:
    """Eagerly initialize the cache (create file + schema) in main process.
    Workers will open their own connections lazily via lookup()/store()."""
    _connect(path)


def canonical_key(reactant_smiles, product_smiles) -> str:
    """Build the canonical cache key. Inputs are iterables of SMILES strings."""
    r = ".".join(sorted(s for s in reactant_smiles if s))
    p = ".".join(sorted(s for s in product_smiles  if s))
    return f"{r}>>{p}"


# Only deterministic mapper outcomes may be persisted. Environmental
# failures (timeouts, OOM, subprocess errors) are machine/load-dependent and
# fail-open in-run; caching them would permanently freeze a transient
# fail-open KEEP as if it were a real verdict.
# VOCABULARY NOTE: forge's aam_filter stores definitive verdicts with status
# 'allowed'/'rejected' (NOT the offline batch mapper's 'mapped'; see the schema
# docstring above); 'no_mapping' is deterministic and cached, but its allowed
# bit is policy-derived, so lookup callers must re-derive the decision from
# the CURRENT keep_on_timeout (aam_filter does).
_CACHEABLE_STATUSES = frozenset({"allowed", "rejected", "no_mapping"})


def lookup(path: str, key: str) -> Optional[Tuple[bool, str]]:
    """Return (allowed, status) or None on miss. Best-effort: treats any
    cache failure (lock timeout, corrupted file) as a miss rather than
    killing the worker."""
    if not path:
        return None
    try:
        con = _connect(path)
        cur = con.execute("SELECT allowed, status FROM aam_cache WHERE rxn_key=?", (key,))
        row = cur.fetchone()
    except Exception:
        return None
    if row is None:
        return None
    return (bool(row[0]), row[1] or "")


def store(path: str, key: str, allowed: bool, status: str, elapsed: float = 0.0) -> None:
    """Write a result. Best-effort: never raises. Environmental statuses
    are not persisted (see _CACHEABLE_STATUSES)."""
    if not path:
        return
    if status not in _CACHEABLE_STATUSES:
        return
    try:
        con = _connect(path)
        con.execute(
            "INSERT OR REPLACE INTO aam_cache(rxn_key, allowed, status, elapsed, ts) "
            "VALUES (?,?,?,?,?)",
            (key, 1 if allowed else 0, status, float(elapsed), time.time()),
        )
    except Exception:
        # cache is best-effort: a write failure must not kill the worker
        pass


def merge_into(dst_path: str, src_path: str) -> int:
    """
    Copy all rows from src cache into dst cache. INSERT OR REPLACE so the
    newer row wins on key conflict. Returns count of rows merged.

    Used to sync a node-local cache back to the persistent cache at
    iter end, and to seed the node-local cache from the persistent cache
    at iter start.
    """
    if not os.path.exists(src_path):
        return 0
    init(dst_path)
    dst = _connect(dst_path)
    try:
        dst.execute("ATTACH DATABASE ? AS src", (src_path,))
        # SOURCE version gate: never bulk-import verdicts computed by a
        # different (or unversioned) mapper version: a raw copy would carry
        # such rows into a freshly version-stamped cache and permanently
        # defeat the _connect gate (both merge directions run every
        # iteration in the two-tier persistent<->node-local flow).
        try:
            srow = dst.execute(
                "SELECT value FROM src.cache_meta WHERE key='mapper_version'"
            ).fetchone()
        except Exception:
            srow = None                    # unversioned cache: no cache_meta
        if srow is None or srow[0] != MAPPER_VERSION:
            try:
                n_src = dst.execute("SELECT count(*) FROM src.aam_cache").fetchone()[0]
            except Exception:
                n_src = 0
            if n_src:
                sv = srow[0] if srow is not None else "<unversioned>"
                print(f"[aam_cache] merge_into: SKIPPING {n_src} rows from "
                      f"{src_path} (source mapper version {sv} != "
                      f"{MAPPER_VERSION})", flush=True)
            dst.execute("DETACH DATABASE src")
            return 0
        dst.execute("BEGIN")
        cur = dst.execute("""
            INSERT OR REPLACE INTO aam_cache(rxn_key, allowed, status, elapsed, ts)
            SELECT rxn_key, allowed, status, elapsed, ts FROM src.aam_cache
        """)
        n = cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0
        dst.execute("COMMIT")
        dst.execute("DETACH DATABASE src")
        return n
    except Exception as e:
        try: dst.execute("ROLLBACK")
        except Exception: pass
        try: dst.execute("DETACH DATABASE src")
        except Exception: pass
        print(f"[aam_cache] merge_into({src_path} -> {dst_path}) failed: {e}", flush=True)
        return 0


def close():
    """Close the per-process connection (call before fork to avoid stale fds)."""
    global _CONN, _PATH
    if _CONN is not None:
        try: _CONN.close()
        except Exception: pass
    _CONN = None
    _PATH = None


def stats(path: str) -> dict:
    """Counts of cached entries, by allowed/disallowed."""
    if not path or not os.path.exists(path):
        return {"total": 0, "allowed": 0, "rejected": 0}
    con = _connect(path)
    total    = con.execute("SELECT count(*) FROM aam_cache").fetchone()[0]
    allowed  = con.execute("SELECT count(*) FROM aam_cache WHERE allowed=1").fetchone()[0]
    rejected = total - allowed
    by_stat  = dict(con.execute(
        "SELECT status, count(*) FROM aam_cache GROUP BY status"
    ).fetchall())
    return {"total": total, "allowed": allowed, "rejected": rejected,
            "by_status": by_stat}
