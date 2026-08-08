"""Per-iteration BATCH AAM filter: decouples atom-mapping from enumeration.

In the decoupled flow (config aam_mode="batch") each iteration runs three phases:
    A) structural enumeration  -> iter_NN/rn_candidates.sqlite  (NO AAM)
    B) batch_aam_filter(...)    -> set of AAM-allowed reaction_ids
    C) append_candidates(...)   -> append allowed rows to the cumulative rn.sqlite

This preserves AAM GATING (AAM runs before kMC every iteration, so flux and
promotion see the AAM-filtered network) while keeping enumeration pure-structural
(MPI-friendly, no inline mapper in the hot loop). AAM verdicts are identical to
the inline path because both call the same aam_check_reaction; batching just
de-duplicates by (reactants, products) first and parallelises across a fork Pool
with the same on-disk cache (persistent across iterations).
"""
from __future__ import annotations

import os
import sqlite3
from typing import Dict, Optional, Set, Tuple

# Reuse the cumulative rn.sqlite schema + writer from the single-node generator.
from forge.seeded_reaction_generator import (
    _CREATE_REACTIONS, _CREATE_METADATA, _CREATE_FACTORS_SHIM, _flush,
    _load_seen_keys,
)


# ---------------------------------------------------------------------------
# Fork-Pool worker (each worker calls the real aam_check_reaction)
# ---------------------------------------------------------------------------
_BG: Dict[str, object] = {}


def _init_worker(mol_entries, cache_path, timeout_s, keep_on_timeout,
                 pre_filter, max_bond_delta):
    _BG["mol_entries"] = mol_entries
    _BG["cache_path"] = cache_path
    _BG["timeout_s"] = timeout_s
    _BG["keep_on_timeout"] = keep_on_timeout
    _BG["pre_filter"] = pre_filter
    _BG["max_bond_delta"] = max_bond_delta


def _worker(reaction):
    from forge.aam_filter import aam_check_reaction
    allowed, _info = aam_check_reaction(
        reaction, _BG["mol_entries"],
        timeout_s=_BG["timeout_s"], keep_on_timeout=_BG["keep_on_timeout"],
        cache_path=_BG["cache_path"], pre_filter=_BG["pre_filter"],
        max_bond_delta=_BG["max_bond_delta"],
    )
    return bool(allowed)


# ---------------------------------------------------------------------------
# Phase B: batch AAM over a candidates DB
# ---------------------------------------------------------------------------
def batch_aam_filter(
    candidates_db: str,
    mol_entries,
    *,
    cache_path: Optional[str],
    timeout_s: int = 120,
    keep_on_timeout: bool = True,
    pre_filter: bool = True,
    max_bond_delta: int = 3,
    nproc: int = 1,
) -> Tuple[Set[int], dict]:
    """Return (set of AAM-allowed reaction_ids, stats).

    Reads every candidate reaction, de-duplicates by (sorted reactants,
    sorted products) so conformer/charge-isomer equivalents collapse, maps each
    UNIQUE reaction once via aam_check_reaction (fork Pool + on-disk cache), then
    expands the verdict back to all candidate ids sharing that key.
    """
    con = sqlite3.connect(candidates_db)
    rows = con.execute(
        "select reaction_id, number_of_reactants, number_of_products, "
        "reactant_1, reactant_2, product_1, product_2 from reactions").fetchall()
    con.close()

    id_to_key: Dict[int, tuple] = {}
    unique: Dict[tuple, dict] = {}
    for (rid, n_r, n_p, r1, r2, p1, p2) in rows:
        reactants = [r1] if n_r == 1 else [r1, r2]
        products = [p1] if n_p == 1 else [p1, p2]
        key = (tuple(sorted(reactants)), tuple(sorted(products)))
        id_to_key[rid] = key
        if key not in unique:
            unique[key] = {"reactants": reactants, "products": products}

    keys = list(unique.keys())
    reactions = [unique[k] for k in keys]

    if not reactions:
        return set(), {"n_candidates": 0, "n_unique": 0, "n_allowed_ids": 0,
                       "n_rejected": 0}

    initargs = (mol_entries, cache_path, timeout_s, keep_on_timeout,
                pre_filter, max_bond_delta)
    if nproc and nproc > 1:
        import multiprocessing as mp
        # Close the module-global cache connection BEFORE forking: the
        # iteration driver warms the cache in this process right before
        # calling us, and forked children would otherwise inherit and reuse
        # the parent's open sqlite fd (via the _connect fast path); sqlite
        # does not support one connection shared across processes.
        # (seeded_reaction_generator closes before forking for this reason.)
        from forge.aam_cache import close as _aam_cache_close
        _aam_cache_close()
        ctx = mp.get_context("fork")
        with ctx.Pool(nproc, initializer=_init_worker, initargs=initargs) as pool:
            verdicts = pool.map(_worker, reactions, chunksize=16)
    else:
        _init_worker(*initargs)
        verdicts = [_worker(r) for r in reactions]

    key_allowed = dict(zip(keys, verdicts))
    allowed_ids = {rid for rid, key in id_to_key.items() if key_allowed[key]}
    stats = {
        "n_candidates": len(rows),
        "n_unique": len(unique),
        "n_allowed_ids": len(allowed_ids),
        "n_rejected": len(rows) - len(allowed_ids),
    }
    return allowed_ids, stats


# ---------------------------------------------------------------------------
# Phase C: append allowed candidate rows to the cumulative rn.sqlite
# ---------------------------------------------------------------------------
def append_candidates(
    candidates_db: str,
    rn_db: str,
    allowed_ids: Optional[Set[int]],
    n_species: int,
    append_mode: bool,
    commit_freq: int = 5000,
) -> dict:
    """Append allowed candidate reactions into the cumulative rn.sqlite with
    fresh dense reaction_ids. allowed_ids=None means keep ALL candidates
    (AAM disabled). Rewrites metadata to the cumulative reaction count."""
    if append_mode and os.path.exists(rn_db):
        con = sqlite3.connect(rn_db)
        con.execute(_CREATE_FACTORS_SHIM)
        mx = con.execute("select max(reaction_id) from reactions").fetchone()[0]
        rid = (mx + 1) if mx is not None else 0
    else:
        if os.path.exists(rn_db):
            os.remove(rn_db)
        con = sqlite3.connect(rn_db)
        con.execute(_CREATE_REACTIONS); con.execute(_CREATE_METADATA)
        con.execute(_CREATE_FACTORS_SHIM); con.commit()
        rid = 0

    seen_keys = _load_seen_keys(con)
    n_dup_skipped = 0

    ccon = sqlite3.connect(candidates_db)
    cur = ccon.execute(
        "select number_of_reactants, number_of_products, reactant_1, reactant_2, "
        "product_1, product_2, rate, dG, dG_barrier, is_redox, reaction_id "
        "from reactions")
    buffer = []
    n_app = 0
    for row in cur:
        fields = row[:10]
        cid = row[10]
        if allowed_ids is not None and cid not in allowed_ids:
            continue
        r1, r2, p1, p2 = fields[2], fields[3], fields[4], fields[5]
        key = ((r1, r2) if r1 <= r2 else (r2, r1),
               (p1, p2) if p1 <= p2 else (p2, p1))
        if key in seen_keys:
            n_dup_skipped += 1
            continue
        seen_keys.add(key)
        buffer.append((rid,) + tuple(fields))
        rid += 1; n_app += 1
        if len(buffer) >= commit_freq:
            _flush(con, buffer); buffer.clear()
    ccon.close()
    if buffer:
        _flush(con, buffer)

    total = con.execute("select count(*) from reactions").fetchone()[0]
    con.execute("DELETE FROM metadata")
    con.execute("INSERT INTO metadata(number_of_species, number_of_reactions) "
                "VALUES (?,?)", (n_species, total))
    con.commit()
    # self-certification: cumulative network must hold no duplicate rows
    dup_groups = con.execute(
        "SELECT COUNT(*) FROM (SELECT 1 FROM reactions "
        "GROUP BY MIN(reactant_1, reactant_2), MAX(reactant_1, reactant_2), "
        "MIN(product_1, product_2), MAX(product_1, product_2) "
        "HAVING COUNT(*) > 1)").fetchone()[0]
    if dup_groups:
        print(f"  [aam_batch] WARNING: {dup_groups} duplicate reaction groups "
              f"in rn.sqlite - dedup guards failed, investigate!", flush=True)
    if n_dup_skipped:
        print(f"  [aam_batch] writer guard skipped {n_dup_skipped} duplicate "
              f"reactions (already in rn.sqlite)", flush=True)
    con.close()
    return {"n_appended": n_app, "n_total": total,
            "n_dup_skipped": n_dup_skipped, "n_duplicate_groups": dup_groups}
