from __future__ import annotations

import argparse
import json
import os
import pickle
import select
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

# Reuse the single-node enumerator's pure pieces unchanged.
from forge.seeded_reaction_generator import (
    build_composition_indexes,
    _init_shared,
    _process_bucket_entry,
    _CREATE_REACTIONS,
    _CREATE_METADATA,
    _CREATE_FACTORS_SHIM,
    _flush,
)
from forge.composition_index_db import (
    DiskCompositionIndex, process_bucket_entry_disk,
)

DISPATCHER_RANK = 0
INITIALIZATION_FINISHED = 0   # worker -> disp, once after index rebuild
SEND_ME_A_WORK_BATCH = 1      # worker -> disp, pull
HERE_IS_A_WORK_BATCH = 2      # disp -> worker, list[task] or None=stop
NEW_REACTION_DB = 3           # worker -> disp, (rows, tested, aam_rej) for one batch

INITIALIZING, RUNNING, FINISHED = 0, 1, 2

_HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_aam_helper_proc.py")


def _log(msg):
    print(f"  [mpi_gen] {msg}", flush=True)

class AamHelper:


    def __init__(self, mol_pickle, cache_path, timeout_s, keep_on_timeout,
                 pre_filter, max_bond_delta):
        self.args = [sys.executable, _HELPER, mol_pickle,
                     cache_path or "", str(timeout_s),
                     "1" if keep_on_timeout else "0",
                     "1" if pre_filter else "0", str(max_bond_delta)]
        self.timeout = timeout_s + 30          # > helper's own hardkill(timeout+5)
        self.keep_on_timeout = keep_on_timeout
        self.spawns = 0
        self.timeouts = 0
        self.proc = None
        self._spawn()

    def _spawn(self):
        # subprocess.Popen with no preexec_fn -> posix_spawn -> fork-free.
        self.proc = subprocess.Popen(
            self.args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self.spawns += 1
        # wait for the helper's READY line (it loads mol_entries first)
        ready, _, _ = select.select([self.proc.stdout], [], [], 600)
        if ready:
            self.proc.stdout.readline()

    def verdict(self, reactants, products):
        req = json.dumps({"reactants": reactants, "products": products})
        try:
            self.proc.stdin.write(req + "\n"); self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            self._spawn()
            return self.keep_on_timeout
        ready, _, _ = select.select([self.proc.stdout], [], [], self.timeout)
        if not ready:
            self.timeouts += 1
            try:
                self.proc.kill(); self.proc.wait()
            except Exception:
                pass
            self._spawn()
            return self.keep_on_timeout
        line = self.proc.stdout.readline().strip()
        if line == "":
            self._spawn()
            return self.keep_on_timeout
        return line == "1"

    def close(self):
        try:
            self.proc.stdin.close(); self.proc.wait(timeout=3)
        except Exception:
            try: self.proc.kill()
            except Exception: pass


# ===========================================================================
#  Dispatcher (rank 0)
# ===========================================================================
def dispatcher(comm, args):
    size = comm.Get_size()
    from mpi4py import MPI

    # --- build the task list (core-filtered), same logic as generate_seeded ---
    core = set(json.loads(Path(args.core_ids_json).read_text()))
    core = set(int(i) for i in core)
    bcon = sqlite3.connect(args.bucket_db)
    bucket_entries = list(bcon.execute("select species_1, species_2 from complexes"))
    bcon.close()
    skip = not args.no_skip_self
    if args.no_enforce_core:
        tasks = [(r1, r2, skip) for (r1, r2) in bucket_entries]
    else:
        tasks = [(r1, r2, skip) for (r1, r2) in bucket_entries
                 if r1 in core or (r2 >= 0 and r2 in core)]
    n_skipped = len(bucket_entries) - len(tasks)
    _log(f"{len(bucket_entries)} bucket entries; {len(tasks)} after core filter "
         f"(skipped {n_skipped}); chunk={args.chunk}")
    # chunk into batches (tiny tuples; dispatcher holds list on ONE rank only)
    batches = [tasks[i:i + args.chunk] for i in range(0, len(tasks), args.chunk)]

    # --- open rn.sqlite (fresh or append) ---
    append = args.append and os.path.exists(args.rn_db)
    if append:
        con = sqlite3.connect(args.rn_db)
        con.execute(_CREATE_FACTORS_SHIM)
        mx = con.execute("select max(reaction_id) from reactions").fetchone()[0]
        reaction_id = (mx + 1) if mx is not None else 0
    else:
        if os.path.exists(args.rn_db):
            os.remove(args.rn_db)
        con = sqlite3.connect(args.rn_db)
        con.execute(_CREATE_REACTIONS); con.execute(_CREATE_METADATA)
        con.execute(_CREATE_FACTORS_SHIM); con.commit()
        reaction_id = 0
    start_id = reaction_id

    seen_keys = set()
    for r1, r2, p1, p2 in con.execute(
            "SELECT reactant_1, reactant_2, product_1, product_2 FROM reactions"):
        seen_keys.add(((r1, r2) if r1 <= r2 else (r2, r1),
                       (p1, p2) if p1 <= p2 else (p2, p1)))
    n_dup_skipped = 0

    worker_states = {r: INITIALIZING for r in range(1, size)}
    for r in range(1, size):
        comm.recv(source=r, tag=INITIALIZATION_FINISHED)
        worker_states[r] = RUNNING
    _log(f"{size-1} workers initialized; dispatching {len(batches)} batches")

    n_tested = n_kept = n_aam_rej = 0
    buffer = []
    status = MPI.Status()
    t0 = time.time()
    while True:
        if RUNNING not in worker_states.values():
            break
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        tag, rank = status.Get_tag(), status.Get_source()
        if tag == SEND_ME_A_WORK_BATCH:
            if batches:
                comm.send(batches.pop(), dest=rank, tag=HERE_IS_A_WORK_BATCH)
            else:
                comm.send(None, dest=rank, tag=HERE_IS_A_WORK_BATCH)
                worker_states[rank] = FINISHED
        elif tag == NEW_REACTION_DB:
            rows, tested, aam_rej = data
            n_tested += tested; n_aam_rej += aam_rej
            for row in rows:                       # row = 10-tuple (no rxn id)
                r = tuple(row)
                key = ((r[2], r[3]) if r[2] <= r[3] else (r[3], r[2]),
                       (r[4], r[5]) if r[4] <= r[5] else (r[5], r[4]))
                if key in seen_keys:
                    n_dup_skipped += 1
                    continue
                seen_keys.add(key)
                buffer.append((reaction_id,) + r)
                reaction_id += 1; n_kept += 1
            if len(buffer) >= args.commit_freq:
                _flush(con, buffer); buffer.clear()
            if n_tested and (n_tested % args.progress_every) < args.chunk:
                dt = time.time() - t0
                _log(f"tested={n_tested} kept={n_kept} aam_rej={n_aam_rej} "
                     f"batches_left={len(batches)} {dt:.0f}s")

    if buffer:
        _flush(con, buffer)
    total_rn = con.execute("select count(*) from reactions").fetchone()[0]
    con.execute("DELETE FROM metadata")
    con.execute("INSERT INTO metadata(number_of_species, number_of_reactions) VALUES (?,?)",
                (args.n_species, total_rn))
    con.commit()
    # self-certification: cumulative network must hold no duplicate rows
    dup_groups = con.execute(
        "SELECT COUNT(*) FROM (SELECT 1 FROM reactions "
        "GROUP BY MIN(reactant_1, reactant_2), MAX(reactant_1, reactant_2), "
        "MIN(product_1, product_2), MAX(product_1, product_2) "
        "HAVING COUNT(*) > 1)").fetchone()[0]
    if dup_groups:
        _log(f"WARNING: {dup_groups} duplicate reaction groups in rn.sqlite "
             f"- dedup guards failed, investigate!")
    else:
        _log("duplicate check: 0 duplicate reaction groups")
    con.close()
    stats = {"n_tested": n_tested, "n_kept": n_kept,
             "n_reactions": reaction_id - start_id, "n_aam_rejected": n_aam_rej,
             "n_dup_skipped": n_dup_skipped, "n_duplicate_groups": dup_groups,
             "n_skipped_nocore": n_skipped, "elapsed_sec": time.time() - t0}
    _log(f"DONE: {n_kept} reactions kept of {n_tested} tested "
         f"(AAM rejected {n_aam_rej}, dup-skipped {n_dup_skipped}) "
         f"in {stats['elapsed_sec']:.0f}s")
    if args.stats_out:
        Path(args.stats_out).write_text(json.dumps(stats))
    print(json.dumps(stats))


# ===========================================================================
#  Worker (rank != 0)
# ===========================================================================
def worker(comm, args):
    rank = comm.Get_rank()
    with open(args.mol_entries, "rb") as f:
        mol_entries = pickle.load(f)
    species_tup = [tuple(sorted(e.species)) for e in mol_entries]
    charges = [e.charge for e in mol_entries]
    G_solv = [e.solvation_free_energy for e in mol_entries]
    params = json.loads(Path(args.params_json).read_text())
    from chemistry_lib.reaction_questions import default_reaction_decision_tree as tree

    disk_index = None
    if args.index_db:
        disk_index = DiskCompositionIndex(args.index_db)
    else:
        unary_by_comp, pair_by_comp = build_composition_indexes(mol_entries)
        _init_shared(mol_entries, tree, params, unary_by_comp, pair_by_comp,
                     species_tup, charges, G_solv, aam_filter=False)

    helper = None
    if args.aam_filter:
        cache = args.aam_cache_path
        if cache:                                  # per-rank cache: no cross-node sqlite
            cache = f"{cache}.rank{rank}"
        helper = AamHelper(args.mol_entries, cache, args.aam_timeout_sec,
                           args.aam_keep_on_timeout, args.aam_pre_filter,
                           args.aam_prefilter_bond_delta_cutoff)
    aam_seen = {}                                  # worker-level exact-dup cache

    comm.send(None, dest=DISPATCHER_RANK, tag=INITIALIZATION_FINISHED)
    while True:
        comm.send(None, dest=DISPATCHER_RANK, tag=SEND_ME_A_WORK_BATCH)
        batch = comm.recv(source=DISPATCHER_RANK, tag=HERE_IS_A_WORK_BATCH)
        if batch is None:
            break
        out_rows = []
        tested = aam_rej = 0
        for task in batch:
            if disk_index is not None:
                kept_rows, t = process_bucket_entry_disk(
                    task, mol_entries, species_tup, charges, G_solv,
                    tree, params, disk_index)
            else:
                kept_rows, t, _ = _process_bucket_entry(task)   # structural only
            tested += t
            for row in kept_rows:
                if helper is not None:
                    n_r, n_p = row[0], row[1]
                    reactants = [row[2]] if n_r == 1 else [row[2], row[3]]
                    products = [row[4]] if n_p == 1 else [row[4], row[5]]
                    key = (tuple(sorted(reactants)), tuple(sorted(products)))
                    if key in aam_seen:
                        keep = aam_seen[key]
                    else:
                        keep = helper.verdict(reactants, products)
                        aam_seen[key] = keep
                    if not keep:
                        aam_rej += 1
                        continue
                out_rows.append(row)
        comm.send((out_rows, tested, aam_rej), dest=DISPATCHER_RANK,
                  tag=NEW_REACTION_DB)

    if helper is not None:
        _log(f"rank {rank} AAM helper: spawns={helper.spawns} "
             f"timeouts={helper.timeouts}")
        helper.close()


# ===========================================================================
def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mol-entries", required=True)
    ap.add_argument("--bucket-db", required=True)
    ap.add_argument("--rn-db", required=True)
    ap.add_argument("--params-json", required=True)
    ap.add_argument("--core-ids-json", required=True)
    ap.add_argument("--index-db", default=None,
                    help="path to a prebuilt composition_index.sqlite; if set, "
                         "workers query it (disk) instead of building pair_by_comp "
                         "in RAM.")
    ap.add_argument("--n-species", type=int, default=0)
    ap.add_argument("--no-enforce-core", action="store_true")
    ap.add_argument("--no-skip-self", action="store_true")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--stats-out", default=None,
                    help="write the dispatcher's final stats JSON to this path")
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--commit-freq", type=int, default=5000)
    ap.add_argument("--progress-every", type=int, default=20000)
    ap.add_argument("--aam-filter", action="store_true")
    ap.add_argument("--aam-timeout-sec", type=int, default=60)
    ap.add_argument("--aam-keep-on-timeout", type=int, default=1)
    ap.add_argument("--aam-cache-path", default=None)
    ap.add_argument("--aam-pre-filter", type=int, default=1)
    ap.add_argument("--aam-prefilter-bond-delta-cutoff", type=int, default=3)
    a = ap.parse_args()
    a.aam_keep_on_timeout = bool(a.aam_keep_on_timeout)
    a.aam_pre_filter = bool(a.aam_pre_filter)
    return a


def _maybe_apply_rq_override():
    """If FORGE_REACTION_QUESTIONS_MODULE is set (inherited from the launching
    job via srun), swap chemistry_lib.reaction_questions for that module BEFORE
    worker() imports default_reaction_decision_tree. A launching process that
    installs an alternate reaction-question module in-process (e.g. a
    paper-equivalent decision tree) must carry that override into the srun'd
    MPI subprocess via this env var; the subprocess starts from a clean
    interpreter and would otherwise use the default tree. No-op when the env
    var is unset (normal runs)."""
    mod = os.environ.get("FORGE_REACTION_QUESTIONS_MODULE")
    if not mod:
        return
    import importlib
    import types as _t
    sys.modules.setdefault("cairo", _t.ModuleType("cairo"))
    import chemistry_lib
    sys.modules.setdefault("HiPRGen", chemistry_lib)
    paper_rq = importlib.import_module(mod)
    sys.modules["chemistry_lib.reaction_questions"] = paper_rq
    try:
        chemistry_lib.reaction_questions = paper_rq
    except Exception:
        pass


def main():
    _maybe_apply_rq_override()
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if size < 2:
        if rank == 0:
            print("forge.mpi_reaction_generator needs >=2 ranks "
                  "(1 dispatcher + >=1 worker). Use mpirun -n N.", file=sys.stderr)
        return
    args = _parse()
    if args.n_species == 0:
        with open(args.mol_entries, "rb") as f:
            args.n_species = len(pickle.load(f))
    if rank == DISPATCHER_RANK:
        dispatcher(comm, args)
    else:
        worker(comm, args)


if __name__ == "__main__":
    main()
