"""Production kMC for a converged FORGE run: consumes the run's config.json
and rn.sqlite (the per-iteration screening kMC uses far fewer trajectories),
runs GMC at high statistics (default 10,000 trajectories x 200,000 steps),
and regenerates sink/tally/species/pathway reports in <out_dir>/<subdir>/.

Invoke: python -m tools.production_kmc --config <out_dir>/config.json
        [--n-simulations N] [--step-cutoff N] [--subdir final_kmc]
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
from pathlib import Path

from forge.run_seeded import (
    _load_config,
    _maybe_run_species_filter,
    _resolve_seeds_from_query,
    _resolve_seeds_from_xyz,
    _validate_seeds,
)
from forge.iteration_driver import _generate_reports, _log
from forge.core_manager import CoreState


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True,
                    help="Path to the FORGE config.json that the run used.")
    ap.add_argument("--n-simulations", type=int, default=10_000,
                    help="kMC trajectories for the production run.")
    ap.add_argument("--step-cutoff", type=int, default=200_000,
                    help="Per-trajectory step cutoff.")
    ap.add_argument("--subdir", default="final_kmc",
                    help="Subdirectory under out_dir to hold the production "
                         "init+trajectories+reports. Default: final_kmc.")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    out = Path(cfg.out_dir)
    rn_db = out / "rn.sqlite"
    state_path = out / "core_state.json"
    if not rn_db.exists():
        raise SystemExit(f"rn.sqlite not found at {rn_db} - has FORGE finished?")
    if not state_path.exists():
        raise SystemExit(f"core_state.json not found at {state_path}")
    state = CoreState.load(state_path)

    # Re-resolve seed species and counts (no-op if seed_species + counts
    # are already set directly in the config).
    _maybe_run_species_filter(cfg)
    _resolve_seeds_from_query(cfg)
    _resolve_seeds_from_xyz(cfg)
    _validate_seeds(cfg)
    _log(f"[production] initial_state_counts: {cfg.initial_state_counts}")

    # Build a fresh production directory + initial state.
    prod_dir = out / args.subdir
    prod_dir.mkdir(parents=True, exist_ok=True)
    prod_init = prod_dir / "initial_state.sqlite"
    if prod_init.exists():
        _log(f"[production] removing existing {prod_init} (will rebuild)")
        prod_init.unlink()

    with open(cfg.mol_entries_pickle, "rb") as f:
        mol_entries = pickle.load(f)

    from chemistry_lib.initial_state import insert_initial_state
    insert_initial_state(cfg.initial_state_counts, mol_entries, str(prod_init))
    _log(f"[production] built fresh {prod_init}")

    # Run GMC against the new init state with the higher trajectory count.
    if not Path(cfg.gmc_binary_path).is_file():
        raise SystemExit(
            f"GMC binary not found at {cfg.gmc_binary_path!r}; check the "
            "config's gmc_binary_path."
        )
    cmd = [
        cfg.gmc_binary_path,
        f"--reaction_database={rn_db}",
        f"--initial_state_database={prod_init}",
        f"--number_of_simulations={args.n_simulations}",
        f"--base_seed={cfg.kmc_seed}",
        f"--thread_count={cfg.gmc_thread_count}",
        f"--step_cutoff={args.step_cutoff}",
    ]
    _log(f"[production] running GMC ({args.n_simulations} sims, "
         f"step_cutoff={args.step_cutoff})")
    rc = subprocess.run(cmd, cwd=str(rn_db.parent)).returncode
    if rc != 0:
        raise SystemExit(f"GMC exited with rc={rc}")
    _log("[production] GMC done")

    # Regenerate reports against the production trajectories. Reports go in
    # prod_dir; mol_pictures gets symlinked from out/mol_pictures.
    _log(f"[production] generating reports in {prod_dir}")
    _generate_reports(cfg, state, init_db=prod_init, reports_dir=prod_dir)
    _log("[production] === done ===")


if __name__ == "__main__":
    main()
