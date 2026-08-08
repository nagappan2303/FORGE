"""
Orchestrates a single iteration of the FORGE loop.

    core_i
      -> seeded bucketing                  (buckets.sqlite: only core-touching pairs)
      -> seeded_reaction_generator         (rn.sqlite via composition-index lookup)
      -> initial_state with seed counts    (initial_state.sqlite)
      -> short kMC (GMC)                   (writes trajectories)
      -> flux extraction                   (per-species produced/consumed counts)
      -> promote species over threshold    (core_{i+1})

Each iteration produces its own folder `iter_{i}/` under out_dir.
After the final iteration converges or hits max_iterations, end-of-run
sink/tally/species/pathway reports are emitted at the run root.
"""
from __future__ import annotations

import json
import os
import pickle
import platform
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from chemistry_lib.initial_state import insert_initial_state

from forge.bucketing_seeded import bucket_seeded, bucket_stats
from forge.flux_extractor import compute_species_flux, promote, summarize
from forge.core_manager import CoreState
from forge.config import SeededConfig
from forge.seeded_reaction_generator import generate_seeded_reactions


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def _log(msg: str):
    print(f"[seeded] {msg}", flush=True)


def _run_enumeration_mpi(cfg, bucket_db, rn_db, params, core_ids,
                         append_mode, aam_cache_active, out,
                         force_structural=False):
    """Launch forge.mpi_reaction_generator across ranks for this iteration's
    enumeration, then return stats shaped like generate_seeded_reactions().

    Params + core_ids go out-of-band as JSON files rather than argv . 
    Progress streams live to our
    stdout/stderr; final stats are read
    from a JSON file the dispatcher writes (avoids buffering a long run's log).
    """
    import shlex
    core_json = Path(out) / "_mpi_core_ids.json"
    params_json = Path(out) / "_mpi_params.json"
    stats_json = Path(out) / "_mpi_stats.json"
    core_json.write_text(json.dumps(sorted(int(i) for i in core_ids)))
    params_json.write_text(json.dumps(params))
    if stats_json.exists():
        stats_json.unlink()

    ranks = cfg.mpi_ranks or int(os.environ.get("SLURM_NTASKS", "0"))
    if ranks < 2:
        raise RuntimeError(
            "enumeration_backend='mpi' needs >=2 ranks (1 dispatcher + >=1 "
            f"worker); got mpi_ranks={cfg.mpi_ranks}, "
            f"SLURM_NTASKS={os.environ.get('SLURM_NTASKS')}. Set cfg.mpi_ranks "
            "or run the job under srun/an MPI allocation.")

    index_args = []
    if cfg.enumeration_index == "disk":
        index_db = Path(cfg.out_dir) / "composition_index.sqlite"
        if not index_db.exists():
            _log("  building on-disk composition index (one-time)...")
            import pickle as _pk
            from forge.composition_index_db import build_index_db
            with open(cfg.mol_entries_pickle, "rb") as _f:
                _me = _pk.load(_f)
            meta = build_index_db(_me, str(index_db))
            _log(f"  composition index built: {meta}")
            del _me
        index_args = ["--index-db", str(index_db)]

    inner = [
        sys.executable, "-m", "forge.mpi_reaction_generator",
        "--mol-entries", cfg.mol_entries_pickle,
        "--bucket-db", str(bucket_db),
        "--rn-db", str(rn_db),
        "--params-json", str(params_json),
        "--core-ids-json", str(core_json),
        "--stats-out", str(stats_json),
        "--commit-freq", str(cfg.commit_freq),
        "--progress-every", str(cfg.progress_every),
    ] + index_args
    if append_mode:
        inner.append("--append")
    if not cfg.enforce_reactant_in_core:
        inner.append("--no-enforce-core")
    if cfg.aam_filter and not force_structural:
        inner += ["--aam-filter",
                  "--aam-timeout-sec", str(cfg.aam_timeout_sec),
                  "--aam-keep-on-timeout", "1" if cfg.aam_keep_on_timeout else "0",
                  "--aam-pre-filter", "1" if cfg.aam_pre_filter else "0",
                  "--aam-prefilter-bond-delta-cutoff",
                  str(cfg.prefilter_bond_delta_cutoff)]
        if aam_cache_active:
            inner += ["--aam-cache-path", str(aam_cache_active)]

    cmd_str = " ".join(shlex.quote(c) for c in inner)
    launch = cfg.mpi_launcher.format(ranks=ranks, cmd=cmd_str)
    _log(f"  enumeration_backend=mpi: launching {ranks} ranks")
    _log(f"  {launch}")
    rc = subprocess.run(launch, shell=True).returncode   # inherit stdout/stderr (live)
    if rc != 0:
        raise RuntimeError(f"MPI enumeration failed (rc={rc})")
    if not stats_json.exists():
        raise RuntimeError(f"MPI enumeration produced no stats at {stats_json}")
    return json.loads(stats_json.read_text())


def _enumerate_then_batch_aam(cfg, mol_entries, tree, params, bucket_db, rn_db,
                              iter_dir, core_ids, aam_cache_active, iter_num, out):
    """Decoupled per-iteration flow (aam_mode='batch'):

       A) structural enumeration  -> iter_NN/rn_candidates.sqlite  (NO AAM)
       B) batch AAM over candidates -> allowed reaction_ids
       C) append allowed rows      -> cumulative rn.sqlite

    AAM still runs before kMC this iteration (gating preserved), but enumeration
    is pure-structural so it can use enumeration_backend='mpi' without an inline
    mapper. Returns stats shaped like generate_seeded_reactions().
    """
    from forge.aam_batch import batch_aam_filter, append_candidates
    import time as _time
    _t0 = _time.time()
    candidates_db = str(Path(iter_dir) / "rn_candidates.sqlite")

    # --- Phase A: structural enumeration into a FRESH per-iter candidates DB ---
    _log("  [batch] Phase A: structural enumeration -> rn_candidates.sqlite")
    if cfg.enumeration_backend == "mpi":
        enum = _run_enumeration_mpi(
            cfg, bucket_db, candidates_db, params, core_ids,
            append_mode=False, aam_cache_active=None, out=out,
            force_structural=True)
    else:
        enum = generate_seeded_reactions(
            mol_entries=mol_entries, bucket_db_file=bucket_db,
            rn_db_file=candidates_db, reaction_decision_tree=tree, params=params,
            core_ids=core_ids, enforce_reactant_in_core=cfg.enforce_reactant_in_core,
            commit_freq=cfg.commit_freq, progress_every=cfg.progress_every,
            nproc=cfg.nproc, chunksize=cfg.pool_chunksize,
            append_mode=False, aam_filter=False)

    # --- Phase B: batch AAM over the unique candidates ---
    allowed = None
    aam_rej = 0
    aam_stats = {}
    if cfg.aam_filter:
        _log("  [batch] Phase B: batch AAM map of candidates")
        allowed, aam_stats = batch_aam_filter(
            candidates_db, mol_entries, cache_path=aam_cache_active,
            timeout_s=cfg.aam_timeout_sec, keep_on_timeout=cfg.aam_keep_on_timeout,
            pre_filter=cfg.aam_pre_filter,
            max_bond_delta=cfg.prefilter_bond_delta_cutoff, nproc=cfg.nproc)
        aam_rej = aam_stats.get("n_rejected", 0)
        _log(f"  [batch] AAM: {aam_stats.get('n_allowed_ids',0)} allowed / "
             f"{aam_rej} rejected of {aam_stats.get('n_candidates',0)} candidates "
             f"({aam_stats.get('n_unique',0)} unique)")

    # --- Phase C: append allowed candidates to the cumulative rn.sqlite ---
    _log("  [batch] Phase C: append allowed -> cumulative rn.sqlite")
    app = append_candidates(candidates_db, rn_db, allowed, len(mol_entries),
                            append_mode=(iter_num > 1),
                            commit_freq=cfg.commit_freq)
    return {"n_tested": enum.get("n_tested", 0),
            "n_kept": app.get("n_appended", 0),
            "n_reactions": app.get("n_appended", 0),
            "n_aam_rejected": aam_rej,
            "n_skipped_nocore": enum.get("n_skipped_nocore", 0),
            "n_structural_candidates": enum.get("n_kept", 0),
            "elapsed_sec": _time.time() - _t0}


def _unlink_if(path):
    p = Path(path)
    if p.exists():
        p.unlink()


def _gmc_runnable(gmc_path: str) -> bool:
    """Best-effort probe: the GMC binary exists, is executable, and starts
    on this host. GMC invoked with no arguments prints its usage text,
    which is enough to prove the binary matches the platform."""
    p = Path(gmc_path)
    if not p.exists():
        return False
    if not os.access(p, os.X_OK):
        return False
    try:
        proc = subprocess.run([str(p)], capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    out = (proc.stdout or b"") + (proc.stderr or b"")
    return proc.returncode == 0 or b"Usage" in out


def _resolve_reaction_tree(name: str = "default"):
    """Return the reaction decision tree referenced by `name`.

    Currently only 'default' is supported (the standard FORGE decision tree
    vendored under chemistry_lib). The `name` argument is kept for forward
    compatibility with future user-supplied trees.
    """
    if name != "default":
        raise ValueError(
            f"unknown reaction_tree={name!r}; only 'default' is supported in FORGE."
        )
    from chemistry_lib.reaction_questions import default_reaction_decision_tree
    return default_reaction_decision_tree


def _trim_rn(rn_db: str, keep_count: int):
    """Delete reactions with reaction_id >= keep_count. Used for resume after
    mid-iter kill so we don't double-append."""
    con = sqlite3.connect(rn_db)
    try:
        # preserve the existing species count: writing 0 would under-size
        # GMC's state arrays if anything consumes the DB before the next
        # generation pass rewrites metadata.
        row = con.execute("SELECT number_of_species FROM metadata").fetchone()
        n_species = row[0] if row and row[0] else 0
        con.execute("DELETE FROM reactions WHERE reaction_id >= ?", (keep_count,))
        con.execute("DELETE FROM metadata")
        n_after = con.execute("SELECT count(*) FROM reactions").fetchone()[0]
        con.execute(
            "INSERT INTO metadata(number_of_species, number_of_reactions) VALUES (?,?)",
            (n_species, n_after),
        )
        con.commit()
    finally:
        con.close()


def count_reactions(rn_db: str) -> int:
    if not Path(rn_db).exists():
        return 0
    con = sqlite3.connect(rn_db)
    try:
        return con.execute("select count(*) from reactions").fetchone()[0]
    finally:
        con.close()


def count_distinct_species_in_reactions(rn_db: str) -> int:
    if not Path(rn_db).exists():
        return 0
    con = sqlite3.connect(rn_db)
    try:
        rows = con.execute(
            "select distinct id from ("
            " select reactant_1 as id from reactions where reactant_1 >= 0"
            " union select reactant_2 as id from reactions where reactant_2 >= 0"
            " union select product_1 as id from reactions where product_1 >= 0"
            " union select product_2 as id from reactions where product_2 >= 0"
            ")"
        ).fetchall()
        return len(rows)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# kMC dispatch
# ---------------------------------------------------------------------------

def run_kmc(cfg: SeededConfig, rn_db: str, initial_state_db: str) -> dict:
    """Run kMC via the compiled GMC binary.

    GMC is the only kMC engine supported. If the binary at
    `cfg.gmc_binary_path` is not runnable on this host (missing file,
    not executable, or non-Linux), this function raises RuntimeError
    with a clear remediation message.
    """
    if not _gmc_runnable(cfg.gmc_binary_path):
        raise RuntimeError(
            f"GMC binary at {cfg.gmc_binary_path!r} is not runnable on this host. "
            "Build GMC for this platform (https://github.com/BlauGroup/RNMC), make "
            "sure it is executable, and set cfg.gmc_binary_path to the absolute "
            "path of the compiled binary."
        )
    _log("running GMC")
    cmd = [
        cfg.gmc_binary_path,
        f"--reaction_database={rn_db}",
        f"--initial_state_database={initial_state_db}",
        f"--number_of_simulations={cfg.n_simulations}",
        f"--base_seed={cfg.kmc_seed}",
        f"--thread_count={cfg.gmc_thread_count}",
        f"--step_cutoff={cfg.step_cutoff}",
    ]

    try:
        probe = subprocess.run([cfg.gmc_binary_path], capture_output=True,
                               timeout=10)
        usage = (probe.stdout or b"") + (probe.stderr or b"")
    except (OSError, subprocess.TimeoutExpired):
        usage = b""
    if b"energy_budget" in usage:
        cmd.append("--energy_budget=0")
    if b"checkpoint" in usage:
        cmd.append("--checkpoint=0")
    completed = subprocess.run(cmd, cwd=str(Path(rn_db).parent))
    return {"backend": "gmc", "rc": completed.returncode}


# ---------------------------------------------------------------------------
# per-iteration step
# ---------------------------------------------------------------------------

def run_one_iteration(
    cfg: SeededConfig,
    mol_entries,
    state: CoreState,
    iter_num: int,
) -> dict:
    """Execute one iteration. Returns a dict with this iteration's stats."""
    out = cfg.out_path()
    iter_dir = out / f"iter_{iter_num:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Cumulative rn.sqlite lives at the run root (NOT per-iter); it grows each iter
    rn_db = str(out / "rn.sqlite")
    bucket_db = str(iter_dir / "buckets.sqlite")
    init_db   = str(iter_dir / "initial_state.sqlite")
    _unlink_if(bucket_db)
    _unlink_if(init_db)

    # --- compute NEW core species for this iter (incremental) ---
    # state.per_iter_promoted[0] = seeds
    # state.per_iter_promoted[k] = species added in iter k
    # After iter_num-1 completes, state.core_ids contains all promoted up to that point.
    # For iter_num=1 the "new" species are the seeds themselves.
    if iter_num == 1:
        new_core_ids = set(int(x) for x in state.core_ids)
    else:
        # species that got promoted in the previous iteration (index iter_num-1)
        last_idx = iter_num - 1
        if last_idx < len(state.per_iter_promoted):
            new_core_ids = set(int(x) for x in state.per_iter_promoted[last_idx])
        else:
            new_core_ids = set()
    _log(f"=== iteration {iter_num} ===  core_size={len(state.core_ids)}  "
         f"new_this_iter={len(new_core_ids)}")

    if not new_core_ids:
        _log("  no new core species to process; skipping reaction generation for this iter")

    # --- 1. seeded bucket (incremental: only new_core-touching pairs) ---
    _log("bucketing (seeded, incremental)...")
    bucket_seeded(mol_entries, bucket_db,
                  core_ids=state.core_ids,
                  new_core_ids=new_core_ids)
    bs = bucket_stats(bucket_db)
    _log(f"  buckets: {bs}")

    # --- 2. seeded reaction generator ---
    _log("generating reactions via composition-index lookup (APPEND to cumulative rn.sqlite)...")
    tree = _resolve_reaction_tree(cfg.reaction_tree)
    params = {
        "temperature": cfg.temperature,
        "electron_free_energy": cfg.electron_free_energy,
    }

    aam_cache_persistent = cfg.aam_cache_path or str(out / "aam_cache.sqlite")
    # Optional two-tier cache: batch scripts can set FORGE_AAM_LOCAL_CACHE_DIR
    # to a node-local path (e.g. /tmp/$USER) to avoid WAL contention on a
    # shared parallel filesystem. The persistent cache is warmed into the
    # local one at iter start and synced back at iter end.
    aam_local_cache_dir = os.environ.get("FORGE_AAM_LOCAL_CACHE_DIR") or None
    if aam_local_cache_dir:
        local_cache_dir = Path(aam_local_cache_dir)
        local_cache_dir.mkdir(parents=True, exist_ok=True)
        aam_cache_active = str(local_cache_dir / "aam_cache.sqlite")
        # Sync persistent -> local at iter start (warm cache from prior runs)
        try:
            from forge.aam_cache import merge_into, stats as _cstats
            n_merged = merge_into(aam_cache_active, aam_cache_persistent)
            cs = _cstats(aam_cache_active)
            _log(f"  [aam_cache] warmed local cache from persistent: "
                 f"merged {n_merged} rows; local now has {cs['total']} entries")
        except Exception as e:
            _log(f"  [aam_cache] warm-from-persistent failed: {e}")
    else:
        aam_cache_active = aam_cache_persistent

    if cfg.aam_mode == "batch":
        # Decoupled per-iteration flow: structural enumerate -> batch AAM ->
        # append. Enumeration stays pure-structural (uses enumeration_backend);
        # AAM is a cached batch pass that still gates this iter's kMC.
        gen_stats = _enumerate_then_batch_aam(
            cfg, mol_entries, tree, params, bucket_db, rn_db, iter_dir,
            state.core_ids, aam_cache_active if cfg.aam_filter else None,
            iter_num, out)
    elif cfg.enumeration_backend == "mpi":
        # Scalable dispatcher-worker enumeration across nodes (OOM-safe).
        # AAM stays inline per rank via a persistent helper; same rn.sqlite.
        gen_stats = _run_enumeration_mpi(
            cfg, bucket_db, rn_db, params, state.core_ids,
            append_mode=(iter_num > 1),
            aam_cache_active=aam_cache_active if cfg.aam_filter else None,
            out=out,
        )
    else:
        gen_stats = generate_seeded_reactions(
            mol_entries=mol_entries,
            bucket_db_file=bucket_db,
            rn_db_file=rn_db,
            reaction_decision_tree=tree,
            params=params,
            core_ids=state.core_ids,
            enforce_reactant_in_core=cfg.enforce_reactant_in_core,
            commit_freq=cfg.commit_freq,
            progress_every=cfg.progress_every,
            nproc=cfg.nproc,
            chunksize=cfg.pool_chunksize,
            append_mode=(iter_num > 1),   # iter 1 creates fresh, iters 2+ append
            aam_filter=cfg.aam_filter,
            aam_timeout_sec=cfg.aam_timeout_sec,
            aam_keep_on_timeout=cfg.aam_keep_on_timeout,
            aam_cache_path=aam_cache_active if cfg.aam_filter else None,
            aam_pre_filter=cfg.aam_pre_filter,
            aam_prefilter_bond_delta_cutoff=cfg.prefilter_bond_delta_cutoff,
            iter_wall_timeout_sec=cfg.iter_wall_timeout_sec,
            iter_progress_log_sec=cfg.iter_progress_log_sec,
        )

    # Sync caches back to the persistent cache at iter end: the node-local
    # cache in the two-tier flow, plus any per-rank caches written by the
    # MPI backend ('{cache}.rankN'), so every rank-computed verdict lands in
    # the persistent cache.
    if cfg.aam_filter:
        try:
            from forge.aam_cache import merge_into, stats as _cstats
            import glob as _glob
            n_merged = 0
            if aam_local_cache_dir:
                n_merged += merge_into(aam_cache_persistent, aam_cache_active)
                for rank_cache in sorted(_glob.glob(f"{aam_cache_active}.rank*")):
                    n_merged += merge_into(aam_cache_persistent, rank_cache)
            for rank_cache in sorted(_glob.glob(f"{aam_cache_persistent}.rank*")):
                n_merged += merge_into(aam_cache_persistent, rank_cache)
            cs = _cstats(aam_cache_persistent)
            _log(f"  [aam_cache] synced to persistent cache: merged {n_merged} "
                 f"rows; persistent cache has {cs['total']} entries")
        except Exception as e:
            _log(f"  [aam_cache] sync-to-persistent failed: {e}")
    n_rxn = count_reactions(rn_db)                      # cumulative total
    n_rxn_new = gen_stats.get("n_reactions", 0)         # this iter only
    n_sp_in_rn = count_distinct_species_in_reactions(rn_db)
    _log(f"  cumulative reactions: {n_rxn}  (+{n_rxn_new} this iter, "
         f"tested={gen_stats.get('n_tested', 0)}, "
         f"elapsed={gen_stats.get('elapsed_sec', 0.0):.1f}s)")

    if n_rxn == 0:
        state.per_iter_bucket_stats.append(bs)
        state.per_iter_rxn_counts.append(0)
        state.per_iter_flux_summary.append({"n_species_touched": 0, "total_events": 0})
        state.iteration = iter_num
        _log(f"  aborting iteration {iter_num}: zero reactions in cumulative rn")
        state.save(out / "core_state.json")
        return {
            "iteration": iter_num, "core_size_in": len(state.core_ids),
            "core_size_out": len(state.core_ids), "bucket_stats": bs,
            "n_reactions": 0, "n_reactions_new_this_iter": 0,
            "n_species_in_rn": 0, "flux_summary": {},
            "kmc_stats": {"backend": "skipped"}, "n_promoted": 0,
            "promoted_ids": [], "error": "zero_rxns", "gen_stats": gen_stats,
        }

    # --- 3. initial state ---
    init_state_counts: Dict[int, int] = dict(cfg.initial_state_counts)

    if cfg.exploratory_initial_count > 0:

        con = sqlite3.connect(rn_db)
        rxn_sp = set()
        for row in con.execute(
            "select reactant_1, reactant_2, product_1, product_2 from reactions"
        ):
            for s in row:
                if s is not None and s >= 0:
                    rxn_sp.add(int(s))
        if cfg.exploratory_only_core_reactants:
            near_core = set()
            ph = ",".join(str(i) for i in state.core_ids)
            for row in con.execute(
                f"select reactant_1, reactant_2, product_1, product_2 from reactions "
                f"where reactant_1 in ({ph}) or reactant_2 in ({ph}) "
                f"   or product_1 in ({ph}) or product_2 in ({ph})"
            ):
                for s in row:
                    if s is not None and s >= 0:
                        near_core.add(int(s))
            rxn_sp = rxn_sp & near_core
        con.close()
        added = 0
        for sid in rxn_sp:
            if sid not in init_state_counts:
                init_state_counts[sid] = cfg.exploratory_initial_count
                added += 1
        _log(f"  exploratory seeding: +{added} species at count={cfg.exploratory_initial_count}")

    insert_initial_state(init_state_counts, mol_entries, init_db)

    # --- 4. short kMC ---
    kmc_stats = run_kmc(cfg, rn_db, init_db)
    _log(f"  kmc: {kmc_stats}")

    # --- 5. flux extraction ---
    flux = compute_species_flux(rn_db, init_db)
    flux_sum = summarize(flux)
    _log(f"  flux summary: {flux_sum}")

    # --- 6. promote ---
    new_ids, promo_cutoff = promote(
        flux,
        existing_core=state.core_ids,
        metric=cfg.promotion_flux_metric,
        fraction=cfg.promotion_flux_fraction,
    )
    _log(f"  promoting {len(new_ids)} new species "
         f"({cfg.promotion_flux_metric} >= {promo_cutoff:.1f} "
         f"[fraction ε={cfg.promotion_flux_fraction:g}])")

    # --- 7. persist ---
    state.iteration = iter_num
    state.per_iter_bucket_stats.append(bs)
    state.per_iter_rxn_counts.append(int(n_rxn_new))   
    state.per_iter_flux_summary.append(flux_sum)
    state.promote(new_ids, iter_num)

    iter_report = {
        "iteration": iter_num,
        "core_size_in": len(state.core_ids) - len(new_ids),
        "core_size_out": len(state.core_ids),
        "new_core_ids_this_iter": sorted(new_core_ids)[:50],
        "bucket_stats": bs,
        "n_reactions_cumulative": int(n_rxn),
        "n_reactions_new_this_iter": int(n_rxn_new),
        "n_species_in_rn_cumulative": int(n_sp_in_rn),
        "flux_summary": flux_sum,
        "kmc_stats": kmc_stats,
        "gen_stats": gen_stats,
        "n_promoted": len(new_ids),
        "promoted_ids": new_ids[:50],
    }
    def _stringify_tuple_keys(o):
        if isinstance(o, dict):
            return {(str(k) if isinstance(k, tuple) else k): _stringify_tuple_keys(v)
                    for k, v in o.items()}
        if isinstance(o, list):
            return [_stringify_tuple_keys(x) for x in o]
        return o
    (iter_dir / "iter_report.json").write_text(
        json.dumps(_stringify_tuple_keys(iter_report), indent=2)
    )
    state.save(out / "core_state.json")

    return iter_report


# ---------------------------------------------------------------------------
# end-of-run report generation
# ---------------------------------------------------------------------------

def _compile_latex(tex_path: Path) -> None:
    """Idempotently compile a .tex file to PDF via pdflatex. Skips if the
    .pdf already exists and is newer than the .tex. Logs and continues
    on failure (pdflatex may not be installed).

    The tex files emit ``\\includegraphics{mol_pictures/<N>.png}`` as relative
    paths. pdflatex resolves those relative to its working directory; the
    folder that contains ``mol_pictures/`` is the run's ``out_dir``, which
    is the tex file's parent for top-level reports, and the tex file's
    grandparent for pathway reports (which sit in ``pathways/``). We invoke
    pdflatex with ``cwd=out_dir`` so both cases resolve correctly.
    """
    pdf_path = tex_path.with_suffix(".pdf")
    if pdf_path.exists() and pdf_path.stat().st_mtime >= tex_path.stat().st_mtime:
        return

    if tex_path.parent.name == "pathways":
        cwd = tex_path.parent.parent
    else:
        cwd = tex_path.parent
    rel_tex = tex_path.relative_to(cwd)

    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode",
             "-output-directory", str(tex_path.parent),
             str(rel_tex)],
            cwd=str(cwd),
            check=False, capture_output=True, timeout=120,
        )
    except FileNotFoundError:
        _log(f"[report] pdflatex not found; leaving {tex_path.name} as .tex only")
    except subprocess.TimeoutExpired:
        _log(f"[report] pdflatex timed out compiling {tex_path.name}")


def _generate_reports(cfg: SeededConfig, state: CoreState,
                      init_db: Optional[Path] = None,
                      reports_dir: Optional[Path] = None) -> None:
    """Build NetworkLoader + SimulationReplayer + Pathfinding from a kMC
    output and emit:
      - <reports_dir>/sink_report.tex
      - <reports_dir>/reaction_tally.tex
      - <reports_dir>/species_report.tex
      - <reports_dir>/pathways/<formula>_<id>.tex for every sink with
        expected_value >= cfg.pathway_ev_cutoff
    Each .tex is auto-compiled via pdflatex when available.

    By default, both ``init_db`` and ``reports_dir`` are inferred from the
    final iteration's per-iter folder (the normal end-of-run path). Callers
    such as the production-kMC tool can override these to point at a
    higher-statistics rerun.
    """
    from chemistry_lib.network_loader import NetworkLoader
    from chemistry_lib import mc_analysis

    out = cfg.out_path()
    final_iter = state.iteration
    iter_dir = out / f"iter_{final_iter:02d}"
    rn_db = out / "rn.sqlite"
    if init_db is None:
        init_db = iter_dir / "initial_state.sqlite"
    if reports_dir is None:
        reports_dir = out
    reports_dir.mkdir(parents=True, exist_ok=True)
    if not (rn_db.exists() and init_db.exists()):
        _log(f"[report] skipping: rn.sqlite or initial_state.sqlite not found "
             f"(rn={rn_db.exists()}, init={init_db.exists()})")
        return

    _log(f"[report] generating end-of-run reports from iter_{final_iter:02d}")
    nl = NetworkLoader(
        network_database=str(rn_db),
        mol_entries_pickle=cfg.mol_entries_pickle,
        initial_state_database=str(init_db),
    )
    nl.load_trajectories()
    nl.load_initial_state()

    # Build mol_pictures/ once in the canonical out_dir before invoking the
    # mc_analysis report functions. Those construct ReportGenerator with
    # rebuild_mol_pictures=False, so each .tex emits
    # \includegraphics{mol_pictures/<idx>.png} that needs the folder to
    # already exist with one image per species index (drawn with RDKit).
    # Idempotent: rebuild only when missing or wrong file count.
    mol_pictures_dir = out / "mol_pictures"
    expected_n = nl.number_of_species
    have = len(list(mol_pictures_dir.glob("*.png"))) if mol_pictures_dir.exists() else 0
    if have != expected_n:
        from chemistry_lib.report_generator import visualize_molecules
        if mol_pictures_dir.exists():
            import shutil
            shutil.rmtree(mol_pictures_dir)
        _log(f"[report] building mol_pictures ({expected_n} species)")
        visualize_molecules(nl.mol_entries, mol_pictures_dir)
        _log(f"[report] mol_pictures built: "
             f"{len(list(mol_pictures_dir.glob('*.png')))} images")
    else:
        _log(f"[report] mol_pictures already present ({have} images)")

    # If reports_dir is a sibling/subdir of out, symlink mol_pictures into
    # it so pdflatex (which we invoke with cwd=reports_dir for top-level
    # .tex files) finds them via the relative path the .tex emits.
    if reports_dir != out:
        sym = reports_dir / "mol_pictures"
        if sym.is_symlink() or sym.exists():
            try:
                if sym.is_symlink():
                    sym.unlink()
                else:
                    import shutil
                    shutil.rmtree(sym)
            except OSError:
                pass
        try:
            sym.symlink_to(mol_pictures_dir, target_is_directory=True)
        except OSError as e:
            _log(f"[report] could not symlink mol_pictures into {reports_dir}: {e}")

    replayer = mc_analysis.SimulationReplayer(nl)
    pf = mc_analysis.Pathfinding(nl)

    tex_files = []
    sink_tex = reports_dir / "sink_report.tex"
    mc_analysis.sink_report(replayer, str(sink_tex), pathfinding=pf); tex_files.append(sink_tex)

    tally_tex = reports_dir / "reaction_tally.tex"
    mc_analysis.reaction_tally_report(nl, str(tally_tex)); tex_files.append(tally_tex)

    species_tex = reports_dir / "species_report.tex"
    mc_analysis.species_report(nl, str(species_tex)); tex_files.append(species_tex)

    # Per-sink pathway reports gated by expected_value cutoff
    pathways_dir = reports_dir / "pathways"; pathways_dir.mkdir(exist_ok=True)
    n_pathway_reports = 0
    for sp_idx in replayer.sinks:
        ev = float(replayer.sink_data[sp_idx].get("expected_value", 0.0))
        if ev < cfg.pathway_ev_cutoff:
            continue
        mol = nl.mol_entries[sp_idx]
        formula = (mol.formula or "spc").replace(" ", "")
        pdf_name = f"{formula}_{sp_idx}.tex"
        path_tex = pathways_dir / pdf_name
        mc_analysis.generate_pathway_report(
            pf, sp_idx, str(path_tex),
            number_of_pathways=cfg.n_pathways_per_sink,
        )
        tex_files.append(path_tex)
        n_pathway_reports += 1
    _log(f"[report] emitted {n_pathway_reports} per-sink pathway reports "
         f"(EV >= {cfg.pathway_ev_cutoff})")

    for tex in tex_files:
        _compile_latex(tex)


# ---------------------------------------------------------------------------
# top-level orchestrator
# ---------------------------------------------------------------------------

def run(cfg: SeededConfig) -> CoreState:
    """Run the full seeded-iterative workflow to convergence or max_iter."""
    with open(cfg.mol_entries_pickle, "rb") as f:
        mol_entries = pickle.load(f)
    _log(f"loaded {len(mol_entries)} mol_entries from {cfg.mol_entries_pickle}")


    out = cfg.out_path()

    # RESUME LOGIC.
    # If core_state.json exists, the previous run got at least to the end of
    # iter 1. Load it, and check whether rn.sqlite is consistent with the
    # last completed iter. If rn.sqlite has MORE rows than expected, the
    # previous run was killed mid-iter; auto-trim back to the last
    # completed iter so we don't double-append.
    state_path = out / "core_state.json"
    rn_path    = out / "rn.sqlite"
    if state_path.exists():
        from forge.core_manager import CoreState as _CS
        state = _CS.load(state_path)
        _log(f"RESUME: loaded core_state.json (iteration={state.iteration}, "
             f"core_size={len(state.core_ids)})")
        # Auto-trim is always on: if a previous run was killed mid-iter, the
        # number of rn.sqlite rows will exceed per_iter_rxn_counts; trim back
        # to the last completed iter so we don't double-append.
        if rn_path.exists():
            expected = sum(int(x) for x in state.per_iter_rxn_counts)
            actual_rn = count_reactions(str(rn_path))
            if actual_rn > expected:
                excess = actual_rn - expected
                _log(f"RESUME: rn.sqlite has {actual_rn} rows but "
                     f"per_iter_rxn_counts sums to {expected}. The previous run "
                     f"was killed mid-iter; trimming {excess} excess rows so "
                     f"the next iter doesn't double-append.")
                _trim_rn(str(rn_path), expected)
                _log(f"RESUME: rn.sqlite trimmed to {count_reactions(str(rn_path))} rows.")
    else:
        state = CoreState()
        state.promote(cfg.seed_species, iteration=0)
        state.save(state_path)

    for it in range(state.iteration + 1, cfg.max_iterations + 1):
        rep = run_one_iteration(cfg, mol_entries, state, it)
        if rep["n_promoted"] <= cfg.convergence_new_species_threshold:
            _log(f"converged at iteration {it}: {rep['n_promoted']} new species "
                 f"<= threshold {cfg.convergence_new_species_threshold}")
            break
    else:
        _log(f"hit max_iterations={cfg.max_iterations} without convergence")

    # final summary
    summary = state.snapshot()
    summary["config"] = {k: str(v) if not isinstance(v, (int, float, str, list, dict, bool, type(None))) else v
                        for k, v in cfg.as_dict().items()}
    (cfg.out_path() / "final_summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"final core size = {len(state.core_ids)}")

    # auto-generate end-of-run reports against the final iteration's kMC
    if cfg.end_of_run_reports:
        try:
            _generate_reports(cfg, state)
        except Exception as e:
            _log(f"[report] auto-report generation failed: {type(e).__name__}: {e}")
    return state
