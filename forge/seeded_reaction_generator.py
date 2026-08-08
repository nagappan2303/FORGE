"""
Seeded reaction generator: composition-lookup enumeration.

Output: rn.sqlite with the standard reaction-network schema.
"""
from __future__ import annotations

import math
import os
import sqlite3
import sys
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple  # noqa: F401

from chemistry_lib.constants import Terminal


# ---------------------------------------------------------------------------
# Composition indexes
# ---------------------------------------------------------------------------

def build_composition_indexes(mol_entries):
    """
    unary_by_comp : tuple[str] -> [ind, ...]
    pair_by_comp  : tuple[str] -> [(i, j), ...]
    """
    n = len(mol_entries)
    species_tup = [tuple(sorted(e.species)) for e in mol_entries]

    unary_by_comp: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for i, e in enumerate(mol_entries):
        unary_by_comp[species_tup[i]].append(e.ind)

    pair_by_comp: Dict[Tuple[str, ...], List[Tuple[int, int]]] = defaultdict(list)
    for i in range(n):
        si = species_tup[i]
        ind_i = mol_entries[i].ind
        for j in range(i, n):
            comb = tuple(sorted(si + species_tup[j]))
            pair_by_comp[comb].append((ind_i, mol_entries[j].ind))

    return unary_by_comp, pair_by_comp


# ---------------------------------------------------------------------------
# Tree walking
# ---------------------------------------------------------------------------

_tree_error_count = 0
_TREE_ERROR_REPORT_LIMIT = 3


def walk_tree(tree, reaction, mol_entries, params) -> Terminal:
    """Walk the reaction decision tree. Returns Terminal.KEEP or Terminal.DISCARD."""
    global _tree_error_count
    for item in tree:
        q, t = item
        try:
            hit = q(reaction, mol_entries, params)
        except Exception:
            _tree_error_count += 1
            if _tree_error_count <= _TREE_ERROR_REPORT_LIMIT:
                import traceback
                print(f"  [seeded_gen] WARNING: decision-tree question "
                      f"{type(q).__name__} raised (error {_tree_error_count}, "
                      f"candidate discarded):", flush=True)
                traceback.print_exc()
                if _tree_error_count == _TREE_ERROR_REPORT_LIMIT:
                    print("  [seeded_gen] further tree-question errors will "
                          "not be reported", flush=True)
            return Terminal.DISCARD
        if hit:
            if isinstance(t, list):
                return walk_tree(t, reaction, mol_entries, params)
            return t
    return Terminal.DISCARD


# ---------------------------------------------------------------------------
# SQL schema
# ---------------------------------------------------------------------------

_CREATE_REACTIONS = """
CREATE TABLE reactions (
    reaction_id INTEGER PRIMARY KEY,
    number_of_reactants INTEGER,
    number_of_products  INTEGER,
    reactant_1 INTEGER, reactant_2 INTEGER,
    product_1  INTEGER, product_2  INTEGER,
    rate REAL, dG REAL, dG_barrier REAL,
    is_redox INTEGER
)
"""
_CREATE_METADATA = """
CREATE TABLE metadata (number_of_species INTEGER, number_of_reactions INTEGER)
"""
_CREATE_FACTORS_SHIM = """
CREATE TABLE IF NOT EXISTS filtered_reactions (
    reaction_id INTEGER, reactant_1 INTEGER, reactant_2 INTEGER,
    product_1 INTEGER, product_2 INTEGER,
    dG REAL, category TEXT
)
"""


# ---------------------------------------------------------------------------
# Rate
# ---------------------------------------------------------------------------

_KB_EV = 8.617333e-5
_H_JS = 6.62607015e-34
_EV_TO_J = 1.602176634e-19


def _eyring_rate(dG_barrier_eV: float, T_K: float) -> float:
    kT = _KB_EV * T_K
    prefactor = (_KB_EV * _EV_TO_J * T_K) / _H_JS   # kT/h in SI
    if dG_barrier_eV <= 0:
        return prefactor
    return prefactor * math.exp(-dG_barrier_eV / kT)


# ---------------------------------------------------------------------------
# Shared state (populated per-process; either main or forked worker)
# ---------------------------------------------------------------------------

_G: Dict[str, object] = {}


def _init_shared(mol_entries, tree, params, unary_by_comp, pair_by_comp,
                  species_tup, charges, G_solv,
                  aam_filter: bool = False,
                  aam_timeout_sec: int = 60,
                  aam_keep_on_timeout: bool = True,
                  aam_cache_path: Optional[str] = None,
                  aam_pre_filter: bool = True,
                  aam_prefilter_bond_delta_cutoff: int = 3):
    _G["mol_entries"] = mol_entries
    _G["tree"] = tree
    _G["params"] = params
    _G["unary_by_comp"] = unary_by_comp
    _G["pair_by_comp"] = pair_by_comp
    _G["species_tup"] = species_tup
    _G["charges"] = charges
    _G["G_solv"] = G_solv
    _G["aam_filter"] = aam_filter
    _G["aam_timeout_sec"] = aam_timeout_sec
    _G["aam_keep_on_timeout"] = aam_keep_on_timeout
    _G["aam_cache_path"] = aam_cache_path
    _G["aam_pre_filter"] = aam_pre_filter
    _G["aam_prefilter_bond_delta_cutoff"] = aam_prefilter_bond_delta_cutoff


def _process_bucket_entry(task):
    """
    Given (r1, r2, skip_self_reactions), enumerate products via composition
    index, tree-walk, and return a list of (kept_row_partial, n_tested) where
    kept_row_partial is a tuple WITHOUT reaction_id (assigned by main later).
    """
    r1, r2, skip_self_reactions = task
    mol_entries      = _G["mol_entries"]
    tree             = _G["tree"]
    params           = _G["params"]
    unary_by_comp    = _G["unary_by_comp"]
    pair_by_comp     = _G["pair_by_comp"]
    species_tup      = _G["species_tup"]
    charges          = _G["charges"]
    G_solv           = _G["G_solv"]
    T                = params.get("temperature", 298.15)
    efe              = params.get("electron_free_energy", 0.0)

    aam_filter           = _G.get("aam_filter", False)
    aam_timeout_sec      = _G.get("aam_timeout_sec", 60)
    aam_keep_on_timeout  = _G.get("aam_keep_on_timeout", True)
    aam_cache_path       = _G.get("aam_cache_path", None)
    aam_pre_filter       = _G.get("aam_pre_filter", True)
    aam_prefilter_bond_delta_cutoff = _G.get("aam_prefilter_bond_delta_cutoff", 3)

    n_r = 1 if r2 == -1 else 2
    if n_r == 1:
        comp = species_tup[r1]
        reactants = [r1]
    else:
        comp = tuple(sorted(list(species_tup[r1]) + list(species_tup[r2])))
        reactants = [r1, r2]

    r_sorted = tuple(sorted(reactants))

    n_tested = 0
    n_aam_rejected = 0
    kept_rows = []   # each row is: (n_r, n_p, r1, r2, p1, p2, rate, dG, dGb, is_redox)

    # unary products
    for p1 in unary_by_comp.get(comp, []):
        products = [p1]
        if skip_self_reactions and tuple(sorted(products)) == r_sorted:
            continue
        reaction = {
            "reactants": reactants, "products": products,
            "number_of_reactants": n_r, "number_of_products": 1,
        }
        n_tested += 1
        if walk_tree(tree, reaction, mol_entries, params) != Terminal.KEEP:
            continue
        # post-tree AAM filter (only if enabled)
        if aam_filter:
            from forge.aam_filter import aam_check_reaction
            allowed, _info = aam_check_reaction(
                reaction, mol_entries,
                timeout_s=aam_timeout_sec,
                keep_on_timeout=aam_keep_on_timeout,
                cache_path=aam_cache_path,
                pre_filter=aam_pre_filter,
                max_bond_delta=aam_prefilter_bond_delta_cutoff,
            )
            if not allowed:
                n_aam_rejected += 1
                continue
        row = _build_row_partial(reactants, products, reaction,
                                 charges, G_solv, efe, T)
        kept_rows.append(row)

    # pair products
    for p1, p2 in pair_by_comp.get(comp, []):
        products = [p1, p2]
        if skip_self_reactions and tuple(sorted(products)) == r_sorted:
            continue
        reaction = {
            "reactants": reactants, "products": products,
            "number_of_reactants": n_r, "number_of_products": 2,
        }
        n_tested += 1
        if walk_tree(tree, reaction, mol_entries, params) != Terminal.KEEP:
            continue
        if aam_filter:
            from forge.aam_filter import aam_check_reaction
            allowed, _info = aam_check_reaction(
                reaction, mol_entries,
                timeout_s=aam_timeout_sec,
                keep_on_timeout=aam_keep_on_timeout,
                cache_path=aam_cache_path,
                pre_filter=aam_pre_filter,
                max_bond_delta=aam_prefilter_bond_delta_cutoff,
            )
            if not allowed:
                n_aam_rejected += 1
                continue
        row = _build_row_partial(reactants, products, reaction,
                                 charges, G_solv, efe, T)
        kept_rows.append(row)

    return (kept_rows, n_tested, n_aam_rejected)


def _build_row_partial(reactants, products, reaction, charges, G_solv, efe, T):
    """Build a tuple (n_r, n_p, r1, r2, p1, p2, rate, dG, dG_barrier, is_redox)."""
    n_r = len(reactants); n_p = len(products)
    r1 = reactants[0]; r2 = reactants[1] if n_r == 2 else -1
    p1 = products[0];  p2 = products[1]  if n_p == 2 else -1
    charge_r = sum(charges[r] for r in reactants)
    charge_p = sum(charges[p] for p in products)
    dcharge  = charge_p - charge_r
    is_redox = 1 if dcharge != 0 else 0
    if "dG" in reaction:
        dG = float(reaction["dG"])
    else:
        dG = sum(G_solv[p] for p in products) - sum(G_solv[r] for r in reactants) \
             + dcharge * float(efe)
    dG_barrier = float(reaction.get("dG_barrier", 0.0))
    rate = float(reaction.get("rate", _eyring_rate(dG_barrier, T)))
    return (n_r, n_p, r1, r2, p1, p2, rate, dG, dG_barrier, is_redox)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_seeded_reactions(
    mol_entries,
    bucket_db_file: str,
    rn_db_file: str,
    reaction_decision_tree,
    params: dict,
    core_ids: Iterable[int],
    enforce_reactant_in_core: bool = True,
    commit_freq: int = 5000,
    progress_every: int = 20000,
    skip_self_reactions: bool = True,
    nproc: int = 1,
    chunksize: int = 32,
    append_mode: bool = False,
    aam_filter: bool = False,
    aam_timeout_sec: int = 60,
    aam_keep_on_timeout: bool = True,
    aam_cache_path: Optional[str] = None,
    aam_pre_filter: bool = True,
    aam_prefilter_bond_delta_cutoff: int = 3,
    iter_wall_timeout_sec: int = 0,
    iter_progress_log_sec: int = 300,
) -> dict:
    """Enumerate & filter reactions for seeded iteration.

    If nproc > 1, uses multiprocessing.Pool with fork start method. On Linux
    the pool's workers inherit mol_entries, tree, indexes via copy-on-write
    memory (no pickling per call), which is critical because mol_entries is
    20-30 MB and the tree has ~10 class instances.
    """
    t0 = time.time()

    print("  [seeded_gen] building composition indexes...", flush=True)
    unary_by_comp, pair_by_comp = build_composition_indexes(mol_entries)
    print(f"  [seeded_gen] indexed {len(unary_by_comp)} unary comps, "
          f"{len(pair_by_comp)} pair comps  ({time.time()-t0:.1f}s)", flush=True)

    species_tup = [tuple(sorted(e.species)) for e in mol_entries]
    charges     = [e.charge for e in mol_entries]
    G_solv      = [e.solvation_free_energy for e in mol_entries]

    core_set = set(int(i) for i in core_ids)

    if aam_filter and aam_cache_path:
        try:
            from forge.aam_cache import init as _aam_cache_init, stats as _aam_cache_stats
            _aam_cache_init(aam_cache_path)
            cs = _aam_cache_stats(aam_cache_path)
            print(f"  [seeded_gen] AAM cache at {aam_cache_path} "
                  f"(existing entries: total={cs['total']} "
                  f"allowed={cs['allowed']} rejected={cs['rejected']})",
                  flush=True)
        except Exception as e:
            print(f"  [seeded_gen] AAM cache init failed: {e}", flush=True)

    if aam_filter and aam_cache_path:
        try:
            from forge.aam_cache import close as _aam_cache_close
            _aam_cache_close()
        except Exception: pass

    _init_shared(mol_entries, reaction_decision_tree, params,
                 unary_by_comp, pair_by_comp, species_tup, charges, G_solv,
                 aam_filter=aam_filter, aam_timeout_sec=aam_timeout_sec,
                 aam_keep_on_timeout=aam_keep_on_timeout,
                 aam_cache_path=aam_cache_path,
                 aam_pre_filter=aam_pre_filter,
                 aam_prefilter_bond_delta_cutoff=aam_prefilter_bond_delta_cutoff)
    if aam_filter:
        print(f"  [seeded_gen] AAM filter ENABLED "
              f"(timeout={aam_timeout_sec}s, "
              f"keep_on_timeout={aam_keep_on_timeout}, "
              f"pre_filter={aam_pre_filter}, "
              f"prefilter_bond_delta_cutoff={aam_prefilter_bond_delta_cutoff}, "
              f"cache={'yes' if aam_cache_path else 'no'})",
              flush=True)
    if iter_wall_timeout_sec > 0:
        print(f"  [seeded_gen] iter wall timeout = {iter_wall_timeout_sec}s "
              f"(finalize-and-continue if exceeded)", flush=True)

    # init rn.sqlite (either fresh or append)
    if append_mode and os.path.exists(rn_db_file):
        con = sqlite3.connect(rn_db_file)
        # make sure schema exists (older DBs might be missing filtered_reactions)
        con.execute(_CREATE_FACTORS_SHIM)
        # determine next reaction_id
        max_id = con.execute("select max(reaction_id) from reactions").fetchone()[0]
        start_id = (max_id + 1) if max_id is not None else 0
        print(f"  [seeded_gen] append mode: existing rn has {max_id+1 if max_id is not None else 0} "
              f"reactions, new IDs start at {start_id}", flush=True)
    else:
        if os.path.exists(rn_db_file):
            os.remove(rn_db_file)
        con = sqlite3.connect(rn_db_file)
        con.execute(_CREATE_REACTIONS)
        con.execute(_CREATE_METADATA)
        con.execute(_CREATE_FACTORS_SHIM)
        con.commit()
        start_id = 0

    # read bucket entries
    bcon = sqlite3.connect(bucket_db_file)
    bucket_entries = list(bcon.execute(
        "select species_1, species_2 from complexes"))
    bcon.close()
    print(f"  [seeded_gen] {len(bucket_entries)} bucket entries to process",
          flush=True)

    if enforce_reactant_in_core:
        filtered = [(r1, r2, skip_self_reactions) for (r1, r2) in bucket_entries
                    if r1 in core_set or (r2 >= 0 and r2 in core_set)]
    else:
        filtered = [(r1, r2, skip_self_reactions) for (r1, r2) in bucket_entries]
    n_skipped_nocore = len(bucket_entries) - len(filtered)
    print(f"  [seeded_gen] {len(filtered)} entries after core filter "
          f"(skipped {n_skipped_nocore} non-core)", flush=True)

    # choose path
    if nproc > 1:
        stats = _run_parallel(
            filtered, con, nproc, chunksize,
            mol_entries, reaction_decision_tree, params,
            unary_by_comp, pair_by_comp,
            species_tup, charges, G_solv,
            commit_freq, progress_every, t0,
            start_reaction_id=start_id,
            iter_wall_timeout_sec=iter_wall_timeout_sec,
            iter_progress_log_sec=iter_progress_log_sec,
            aam_cache_path=aam_cache_path if aam_filter else None,
        )
    else:
        stats = _run_sequential(filtered, con, commit_freq, progress_every, t0,
                                start_reaction_id=start_id)

    # metadata: total reactions in this DB (cumulative if append_mode)
    total_rn = con.execute("select count(*) from reactions").fetchone()[0]
    con.execute("DELETE FROM metadata")
    con.execute("INSERT INTO metadata(number_of_species, number_of_reactions) VALUES (?, ?)",
                (len(mol_entries), total_rn))
    con.commit()

    # self-certification: the cumulative network must contain no duplicate
    # (unordered reactants -> unordered products) rows. The bucketing guard
    # plus the writer seen-set should make this impossible;
    dup_groups = con.execute(
        "SELECT COUNT(*) FROM (SELECT 1 FROM reactions "
        "GROUP BY MIN(reactant_1, reactant_2), MAX(reactant_1, reactant_2), "
        "MIN(product_1, product_2), MAX(product_1, product_2) "
        "HAVING COUNT(*) > 1)").fetchone()[0]
    if dup_groups:
        print(f"  [seeded_gen] WARNING: {dup_groups} duplicate reaction groups "
              f"in rn.sqlite - dedup guards failed, investigate!", flush=True)
    else:
        print("  [seeded_gen] duplicate check: 0 duplicate reaction groups",
              flush=True)
    stats["n_duplicate_groups"] = dup_groups
    con.close()

    stats["n_skipped_nocore"] = n_skipped_nocore
    stats["elapsed_sec"] = time.time() - t0
    aam_msg = ""
    if aam_filter:
        aam_msg = f" (AAM rejected {stats.get('n_aam_rejected', 0)})"
    print(f"  [seeded_gen] done: {stats['n_kept']} reactions kept "
          f"out of {stats['n_tested']} tested{aam_msg} "
          f"({stats['elapsed_sec']:.1f}s)", flush=True)
    return stats


# ---------------------------------------------------------------------------
# Sequential path
# ---------------------------------------------------------------------------

def _run_sequential(filtered_entries, con, commit_freq, progress_every, t0,
                    start_reaction_id: int = 0):
    n_tested = 0; n_kept = 0; reaction_id = start_reaction_id
    n_aam_rejected = 0
    n_dup_skipped = 0
    last_progress_n = 0
    by_rcount = defaultdict(int)
    seen_keys = _load_seen_keys(con)
    buffer = []
    for task in filtered_entries:
        kept_rows, tested, aam_rej = _process_bucket_entry(task)
        n_tested += tested
        n_aam_rejected += aam_rej
        for row in kept_rows:
            key = _canon_key(row)
            if key in seen_keys:
                n_dup_skipped += 1
                continue
            seen_keys.add(key)
            buffer.append((reaction_id,) + row)
            reaction_id += 1
            n_kept += 1
            by_rcount[f"{row[0]}->{row[1]}"] += 1
        if len(buffer) >= commit_freq:
            _flush(con, buffer); buffer.clear()
        if n_tested - last_progress_n >= progress_every:
            dt = time.time() - t0
            print(f"  [seeded_gen] tested={n_tested:>10} kept={n_kept:>7} "
                  f"elapsed={dt:.0f}s  rate={n_tested/max(1,dt):.0f}/s",
                  flush=True)
            last_progress_n = n_tested
    if buffer:
        _flush(con, buffer)
    if n_dup_skipped:
        print(f"  [seeded_gen] writer guard skipped {n_dup_skipped} duplicate "
              f"reactions (already in rn.sqlite)", flush=True)
    return {"n_tested": n_tested, "n_kept": n_kept,
            "n_reactions": reaction_id - start_reaction_id,
            "n_aam_rejected": n_aam_rejected,
            "n_dup_skipped": n_dup_skipped,
            "by_rcount": dict(by_rcount)}


# ---------------------------------------------------------------------------
# Parallel path (fork-based multiprocessing)
# ---------------------------------------------------------------------------

def _run_parallel(filtered_entries, con, nproc, chunksize,
                  mol_entries, tree, params, unary_by_comp, pair_by_comp,
                  species_tup, charges, G_solv,
                  commit_freq, progress_every, t0,
                  start_reaction_id: int = 0,
                  iter_wall_timeout_sec: int = 14400,
                  iter_progress_log_sec: int = 300,
                  aam_cache_path: Optional[str] = None):
    """Wall-timeout aware + AAM-rate watchdog.

    If iter_wall_timeout_sec > 0 and the iter exceeds that wall time during
    reaction generation, we stop accepting new tasks, drain the in-flight
    chunks, flush the buffer, and return. The caller still runs kMC on whatever
    AAM-clean reactions made it into rn.sqlite, i.e. the iter is finalized
    early instead of hanging forever.
    """
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    n_tested = 0; n_kept = 0; reaction_id = start_reaction_id
    n_aam_rejected = 0
    n_dup_skipped = 0
    by_rcount = defaultdict(int)
    seen_keys = _load_seen_keys(con)
    buffer = []
    last_progress_n = 0
    last_log_t = time.time()
    last_aam_count = _get_aam_count(aam_cache_path)
    early_finalized = False
    n_done = 0
    n_total = len(filtered_entries)

    # Fork pool AFTER shared state is set
    with ctx.Pool(nproc) as pool:
        print(f"  [seeded_gen] parallel mode: {nproc} workers, chunksize={chunksize}", flush=True)
        result_iter = pool.imap_unordered(
            _process_bucket_entry, filtered_entries, chunksize=chunksize
        )

        for kept_rows, tested, aam_rej in result_iter:
            n_tested += tested
            n_aam_rejected += aam_rej
            n_done += 1
            for row in kept_rows:
                key = _canon_key(row)
                if key in seen_keys:
                    n_dup_skipped += 1
                    continue
                seen_keys.add(key)
                buffer.append((reaction_id,) + row)
                reaction_id += 1
                n_kept += 1
                by_rcount[f"{row[0]}->{row[1]}"] += 1
            if len(buffer) >= commit_freq:
                _flush(con, buffer); buffer.clear()

            now = time.time()

            # routine progress (every progress_every tested)
            if n_tested - last_progress_n >= progress_every:
                dt = now - t0
                print(f"  [seeded_gen] tested={n_tested:>10} kept={n_kept:>7} "
                      f"aam_rej={n_aam_rejected:>6} bucket_done={n_done}/{n_total} "
                      f"elapsed={dt:.0f}s  rate={n_tested/max(1,dt):.0f}/s",
                      flush=True)
                last_progress_n = n_tested

            # wall-clock watchdog (every iter_progress_log_sec)
            if now - last_log_t >= iter_progress_log_sec:
                dt = now - t0
                cur_aam = _get_aam_count(aam_cache_path)
                d_aam = cur_aam - last_aam_count
                aam_rate = d_aam / (now - last_log_t)
                print(f"  [seeded_gen][watchdog] elapsed={dt:.0f}s "
                      f"buckets_done={n_done}/{n_total} ({100*n_done/max(1,n_total):.1f}%) "
                      f"AAM_decisions_recent={d_aam} ({aam_rate:.2f}/s) "
                      f"AAM_total={cur_aam}", flush=True)
                if aam_rate < 0.05 and cur_aam > 0:
                    print(f"  [seeded_gen][watchdog] WARNING: AAM rate is "
                          f"very low ({aam_rate:.3f}/s). Likely a stall.",
                          flush=True)
                last_aam_count = cur_aam
                last_log_t = now

            # iter wall timeout: finalize-and-continue
            if iter_wall_timeout_sec > 0 and (now - t0) > iter_wall_timeout_sec:
                dt = now - t0
                print(f"  [seeded_gen][TIMEOUT] iter exceeded "
                      f"{iter_wall_timeout_sec}s ({dt:.0f}s elapsed); "
                      f"finalizing with {n_kept} reactions kept "
                      f"({n_done}/{n_total} bucket entries done = "
                      f"{100*n_done/max(1,n_total):.1f}%). Terminating pool.",
                      flush=True)
                early_finalized = True
                pool.terminate()
                pool.join()
                break

    if buffer:
        _flush(con, buffer)
    if n_dup_skipped:
        print(f"  [seeded_gen] writer guard skipped {n_dup_skipped} duplicate "
              f"reactions (already in rn.sqlite)", flush=True)
    return {"n_tested": n_tested, "n_kept": n_kept,
            "n_reactions": reaction_id - start_reaction_id,
            "n_aam_rejected": n_aam_rejected,
            "n_dup_skipped": n_dup_skipped,
            "n_buckets_done": n_done, "n_buckets_total": n_total,
            "early_finalized": early_finalized,
            "by_rcount": dict(by_rcount)}


def _canon_key(row):
    """Canonical (unordered reactants, unordered products) key of a kept row.

    row = (number_of_reactants, number_of_products, reactant_1, reactant_2,
           product_1, product_2, rate, dG, dG_barrier, is_redox)
    Direction is preserved (reverse reactions have swapped sides and remain
    distinct keys).
    """
    r1, r2, p1, p2 = row[2], row[3], row[4], row[5]
    return ((r1, r2) if r1 <= r2 else (r2, r1),
            (p1, p2) if p1 <= p2 else (p2, p1))


def _load_seen_keys(con):
    """Canonical keys of every reaction already in rn.sqlite.

    Writer-level duplicate guard (belt-and-braces on top of the incremental
    bucketing guard): protects append-mode restarts and any future emission
    bug from silently doubling kMC propensities."""
    seen = set()
    for r1, r2, p1, p2 in con.execute(
            "SELECT reactant_1, reactant_2, product_1, product_2 FROM reactions"):
        seen.add(((r1, r2) if r1 <= r2 else (r2, r1),
                  (p1, p2) if p1 <= p2 else (p2, p1)))
    return seen


def _get_aam_count(cache_path):
    """Quick read of cache row count for watchdog. Returns 0 on any error."""
    if not cache_path or not os.path.exists(cache_path):
        return 0
    try:
        c = sqlite3.connect(cache_path, timeout=2)
        n = c.execute("SELECT count(*) FROM aam_cache").fetchone()[0]
        c.close()
        return n
    except Exception:
        return 0


def _flush(con, buffer):
    con.executemany(
        "INSERT INTO reactions(reaction_id, number_of_reactants, number_of_products, "
        "reactant_1, reactant_2, product_1, product_2, rate, dG, dG_barrier, is_redox) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        buffer,
    )
    con.commit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse, pickle, json
    from monty.serialization import loadfn
    ap = argparse.ArgumentParser()
    ap.add_argument("--mol-entries", required=True)
    ap.add_argument("--bucket-db", required=True)
    ap.add_argument("--rn-db", required=True)
    ap.add_argument("--params-json", required=True)
    ap.add_argument("--core-ids", required=True)
    ap.add_argument("--tree", default="default")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--no-enforce-core", action="store_true")
    args = ap.parse_args()

    with open(args.mol_entries, "rb") as f:
        mol_entries = pickle.load(f)
    try:    params = loadfn(args.params_json)
    except: params = json.load(open(args.params_json))

    if args.tree != "default":
        raise ValueError(
            f"unknown --tree={args.tree!r}; only 'default' is supported in FORGE."
        )
    from chemistry_lib.reaction_questions import default_reaction_decision_tree as tree

    core_ids = [int(x) for x in args.core_ids.split(",") if x.strip()]

    stats = generate_seeded_reactions(
        mol_entries=mol_entries,
        bucket_db_file=args.bucket_db,
        rn_db_file=args.rn_db,
        reaction_decision_tree=tree,
        params=params,
        core_ids=core_ids,
        enforce_reactant_in_core=not args.no_enforce_core,
        nproc=args.nproc,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    _cli()
