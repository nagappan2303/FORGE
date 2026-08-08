"""
Self-contained FORGE example: Li+ / EC electrolyte decomposition.

Runs the complete FORGE loop, seeded bucketing, reaction enumeration,
decision-tree filtering, the inline atom-to-atom mapping filter,
screening kMC, and flux-based promotion, on the bundled test pool
(data/li_ec_test.pickle, 608 C/H/O/Li species drawn from the production
Li/EC dataset of the FORGE paper). The loop stops on its own when an
iteration promotes no new species; expect roughly 5-15 minutes on a
laptop. The run ends with a summary of the generated network; for
product and pathway reports on a converged network, see
tools/production_kmc.py.

Requirements
------------
1. Python dependencies of this repository (see pyproject.toml).
2. The GMC kinetic Monte Carlo binary from BlauGroup/RNMC. Build it once:

       git clone https://github.com/BlauGroup/RNMC
       cd RNMC/GMC && mkdir -p ../build && make GMC

   and pass the binary path with --gmc (or set the GMC_BIN environment
   variable). The build needs a C++17 compiler, GSL, and SQLite3.

Usage
-----
    python examples/run_li_ec_demo.py --gmc /path/to/RNMC/build/GMC

The promotion threshold epsilon (Eq. 1 of the paper) defaults to 1e-2
here; pass --eps to explore other values (smaller epsilon grows a larger
network and takes longer). The AAM verdict cache in the output directory
makes repeated runs faster.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

# allow running the example straight from a source checkout, with or
# without pip-installing the package
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from forge.config import SeededConfig
from forge.iteration_driver import run

# silence Open Babel's harmless bond-perception warnings
try:
    from openbabel import openbabel as _ob
    _ob.obErrorLog.SetOutputLevel(0)
except Exception:
    pass

HERE = Path(__file__).resolve().parent


def find_lowest_G(entries, formula: str, charge: int):
    """Lowest solvation-free-energy entry matching formula and charge."""
    cands = [
        e for e in entries
        if e.formula.replace(" ", "") == formula.replace(" ", "")
        and e.charge == charge
    ]
    if not cands:
        return None
    return min(cands, key=lambda e: e.solvation_free_energy)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--gmc", default=os.environ.get("GMC_BIN", ""),
                    help="path to the compiled GMC binary (or set GMC_BIN)")
    ap.add_argument("--out", default=str(HERE / "out_li_ec"),
                    help="output directory")
    ap.add_argument("--eps", type=float, default=1e-2,
                    help="normalized flux promotion threshold (default 1e-2)")
    args = ap.parse_args()

    if not args.gmc or not Path(args.gmc).exists():
        raise SystemExit(
            "GMC binary not found. Build it from BlauGroup/RNMC (see the "
            "docstring at the top of this file) and pass --gmc /path/to/GMC "
            "or set the GMC_BIN environment variable.")
    # GMC is invoked from inside the per-iteration folders, so relative
    # paths must be made absolute here
    args.gmc = str(Path(args.gmc).expanduser().resolve())
    args.out = str(Path(args.out).expanduser().resolve())

    data = str(HERE / "data" / "li_ec_test.pickle")
    t0 = time.time()
    with open(data, "rb") as f:
        pool = pickle.load(f)
    print(f"[demo] species pool: {len(pool)} species from {data}")

    Li_plus = find_lowest_G(pool, "Li1", +1)
    EC = find_lowest_G(pool, "C3H4O3", 0)
    if Li_plus is None or EC is None:
        raise SystemExit("seed species (Li+, EC) not found in the pool")
    print(f"[demo] seeds: Li+ ind={Li_plus.ind}, EC ind={EC.ind}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SeededConfig(
        mol_entries_pickle=data,
        seed_species=[Li_plus.ind, EC.ind],
        initial_state_counts={Li_plus.ind: 100, EC.ind: 1500},
        out_dir=str(out_dir),
        electron_free_energy=-1.40,
        max_iterations=12,
        promotion_flux_fraction=args.eps,
        promotion_flux_metric="produced",
        convergence_new_species_threshold=0,
        n_simulations=50,
        step_cutoff=20000,
        gmc_binary_path=args.gmc,
        gmc_thread_count=max(2, (os.cpu_count() or 4) - 2),
        aam_filter=True,
        enforce_reactant_in_core=True,
        # single node, single machine: the example never needs the cluster
        # enumeration backend
        enumeration_backend="fork",
        aam_mode="inline",
        # keep the example lean: no end-of-run report generation
        end_of_run_reports=False,
    )

    print(f"[demo] running FORGE (eps={args.eps}, AAM=on) ...")
    state = run(cfg)

    elapsed = time.time() - t0

    import sqlite3
    n_rxn = sqlite3.connect(str(out_dir / "rn.sqlite")).execute(
        "select count(*) from reactions").fetchone()[0]
    aam_ok = aam_rej = 0
    cache = out_dir / "aam_cache.sqlite"
    if cache.exists():
        for status, n in sqlite3.connect(str(cache)).execute(
                "select status, count(*) from aam_cache group by status"):
            if status == "allowed":
                aam_ok = n
            elif status == "rejected":
                aam_rej = n

    print(f"\n[demo] DONE in {elapsed/60:.1f} min")
    print(f"  iterations run:       {state.iteration}")
    print(f"  final core size:      {len(state.core_ids)}")
    print(f"  promoted per iter:    {[len(x) for x in state.per_iter_promoted]}")
    print(f"  reactions per iter:   {state.per_iter_rxn_counts}")
    print(f"  network:              {n_rxn} reactions -> rn.sqlite")
    print(f"  atom-mapping:         {aam_ok} unique reactions verified "
          f"elementary, {aam_rej} rejected")
    print(f"\n[demo] outputs in {out_dir}: rn.sqlite, final_summary.json, "
          f"per-iteration folders")


if __name__ == "__main__":
    main()
