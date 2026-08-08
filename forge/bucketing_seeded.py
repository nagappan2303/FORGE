"""
Seeded bucketing: emits pair buckets where at least one species is in
the supplied `core_ids` set.

This is the single algorithmic change that shifts CRN construction from
exhaustive O(N^2) pair enumeration to O(|core| * N) per iteration.
"""
from __future__ import annotations

import sqlite3
from typing import Iterable, Optional, Sequence, Set


def bucket_seeded(
    mol_entries,
    bucket_db_path: str,
    core_ids: Iterable[int],
    new_core_ids: Optional[Iterable[int]] = None,
    commit_freq: int = 2000,
    group_size: int = 1000,
    also_include_unary: bool = True,
    dense_pairs_in_core: bool = True,
):
    """Write a seeded bucket database.

    Two modes:

    - **Full mode** (new_core_ids is None): emit pairs where >=1 species is in
      `core_ids`. Use for one-off runs or iter 1 when nothing has been
      processed yet.

    - **Incremental mode** (new_core_ids is a set): emit only pairs where >=1
      species is in `new_core_ids`, the subset of core_ids that was added
      THIS iteration. Pairs already processed in prior iterations (both
      species were in pre-iter core) are skipped. This prevents re-walking
      the same reaction through the tree.

    Parameters
    ----------
    mol_entries : list of MoleculeEntry
    bucket_db_path : str
    core_ids : iterable of int
        All species currently in core. Used to enforce seeded discipline.
    new_core_ids : iterable of int or None
        If given, only emit pairs touching one of these. If None, emit
        pairs touching any core_ids species (full mode).
    also_include_unary : bool
        Emit (species, -1) single-species buckets. In incremental mode,
        only for species in new_core_ids (their unary reactions haven't
        been enumerated yet).
    dense_pairs_in_core : bool
        Emit intra-core pairs (both species in core). Always True in practice.
    """
    core = set(int(i) for i in core_ids)
    new_core = set(int(i) for i in new_core_ids) if new_core_ids is not None else None
    incremental = new_core is not None
    # Species already in core BEFORE this iteration. Pairs (new, old_core)
    # were already emitted in the iteration the old member entered the core
    # (it was bucketed against the FULL pool then), so re-emitting them here
    # duplicates reactions in the cumulative rn.sqlite.
    old_core = (core - new_core) if incremental else set()

    con = sqlite3.connect(bucket_db_path)
    cur = con.cursor()
    cur.execute(
        "CREATE TABLE complexes (species_1, species_2, composition_id, group_id)"
    )
    cur.execute(
        "CREATE INDEX composition_index ON complexes (composition_id, group_id)"
    )

    group_counts: dict = {}
    bucket_counts: dict = {}
    composition_ids: dict = {}
    commit_count = 0
    composition_count = 0

    def _insert(species_1, species_2, composition_key):
        nonlocal composition_count, commit_count
        if composition_key not in group_counts:
            group_counts[composition_key] = 0
            bucket_counts[composition_key] = 0
            composition_ids[composition_key] = composition_count
            composition_count += 1
        cur.execute(
            "INSERT INTO complexes VALUES (?, ?, ?, ?)",
            (species_1,
             species_2,
             composition_ids[composition_key],
             group_counts[composition_key]),
        )
        commit_count += 1
        if commit_count % commit_freq == 0:
            con.commit()
        bucket_counts[composition_key] += 1
        if bucket_counts[composition_key] % group_size == 0:
            group_counts[composition_key] += 1

    # unary buckets
    if also_include_unary:
        # incremental: only new_core species need their unary processed this
        # iter.
        # full mode: every core species.
        entries_by_ind = {m.ind: m for m in mol_entries}
        for ind in (new_core if incremental else core):
            m = entries_by_ind[ind]
            comp = '_'.join(sorted(m.species))
            _insert(m.ind, -1, comp)

    # pair buckets
    n = len(mol_entries)
    for i in range(n):
        m1 = mol_entries[i]
        for j in range(i, n):
            m2 = mol_entries[j]
            if incremental:
                # Incremental: emit only pairs that touch new_core AND do not
                # touch old_core. Pairs with both members in pre-iter core
                # were emitted in earlier iterations; pairs (new, old_core)
                # were also already emitted, in the iteration the old member
                # entered the core, when it was paired against the full pool.
                i_new = (m1.ind in new_core)
                j_new = (m2.ind in new_core)
                if not (i_new or j_new):
                    continue
                if i_new != j_new and ((m2.ind if i_new else m1.ind) in old_core):
                    continue
            else:
                i_in = (m1.ind in core)
                j_in = (m2.ind in core)
                if not (i_in or j_in):
                    continue
                if (i_in and j_in) and not dense_pairs_in_core:
                    continue
            comp = '_'.join(sorted(m1.species + m2.species))
            _insert(m1.ind, m2.ind, comp)

    # tail tables that the reaction-filter pipeline reads
    con.execute("CREATE TABLE group_counts (composition_id, count)")
    con.execute("CREATE TABLE compositions (composition_id, composition)")
    for composition, cid in composition_ids.items():
        cur.execute(
            "INSERT INTO group_counts VALUES (?, ?)",
            (cid, group_counts[composition] + 1),
        )
        cur.execute(
            "INSERT INTO compositions VALUES (?, ?)",
            (cid, composition),
        )

    con.commit()
    con.close()


def bucket_stats(bucket_db_path: str) -> dict:
    """Return summary of a bucket DB for logging."""
    con = sqlite3.connect(bucket_db_path)
    cur = con.cursor()
    n_rows = cur.execute("select count(*) from complexes").fetchone()[0]
    n_unary = cur.execute("select count(*) from complexes where species_2 = -1").fetchone()[0]
    n_pair = n_rows - n_unary
    n_comp = cur.execute("select count(*) from compositions").fetchone()[0]
    con.close()
    return {
        "n_rows": n_rows,
        "n_unary": n_unary,
        "n_pair": n_pair,
        "n_compositions": n_comp,
    }
