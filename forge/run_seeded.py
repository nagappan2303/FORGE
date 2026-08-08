"""CLI entry point for FORGE seeded-iterative runs.

Usage:
    python -m forge.run_seeded --config run_config.json

The JSON config maps one-to-one onto fields in forge.config.SeededConfig;
every field is documented inline in forge/config.py.

Seed-species specification modes (pick exactly one):

1. Direct indices: set `seed_species` and `initial_state_counts` to the
   integer indices in the pickle. Brittle: indices change if the pickle
   is rebuilt.

2. By formula+charge (`seed_species_query`). Each entry is
   {"formula": "Li1", "charge": 1, "count": 600}. FORGE picks the
   lowest-G coordimer for each (formula, charge) tuple from the pickle
   after species_filter runs, and populates seed_species +
   initial_state_counts automatically.

3. By XYZ file (`seed_species_xyz`): for species whose canonical
   formula+charge is ambiguous. Each entry is {"xyz": "/path", "charge":
   0, "spin": 1, "count": 9000}.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

from forge.config import SeededConfig
from forge.iteration_driver import run


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config(path: str) -> SeededConfig:
    raw = json.loads(Path(path).read_text())
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    # JSON dict keys are strings; SeededConfig expects int species ids
    isc = raw.get("initial_state_counts", {})
    if isc:
        raw["initial_state_counts"] = {int(k): int(v) for k, v in isc.items()}

    known = set(SeededConfig.__dataclass_fields__)
    unknown = sorted(k for k in raw if k not in known)
    if unknown:
        print(f"[config] ignoring unknown keys: {', '.join(unknown)}")
        raw = {k: v for k, v in raw.items() if k in known}
    return SeededConfig(**raw)



def _maybe_run_species_filter(cfg: SeededConfig) -> None:
    """If `species_json_path` is set and the pickle does not yet exist, run
    chemistry_lib.species_filter to produce it. Idempotent: a second
    invocation against the same pickle path skips the work.
    """
    if not cfg.species_json_path:
        return
    pickle_path = Path(cfg.mol_entries_pickle)
    if pickle_path.exists():
        print(f"[forge] mol_entries pickle already present at {pickle_path}; "
              f"skipping species_filter.")
        return
    if not cfg.solvation_environment:
        raise ValueError(
            "species_json_path is set but solvation_environment is not. "
            "Pick one of the supported entries in chemistry_lib.constants "
            "('li_ec' or 'Na_ec') to select the solvation reference."
        )
    from chemistry_lib import constants as _constants
    from chemistry_lib.species_filter import species_filter
    from chemistry_lib.species_questions import make_species_decision_tree

    solv_env = getattr(_constants, cfg.solvation_environment, None)
    if solv_env is None:
        raise ValueError(
            f"Unknown solvation_environment={cfg.solvation_environment!r}. "
            f"FORGE currently supports 'li_ec' and 'Na_ec' (see "
            f"chemistry_lib/constants.py)."
        )

    tree = make_species_decision_tree(solv_env)

    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    species_report_path = str(out_dir / "species_filter_report.tex")

    from monty.serialization import loadfn
    dataset_entries = loadfn(cfg.species_json_path)
    print(f"[forge] running species_filter on {cfg.species_json_path} "
          f"({len(dataset_entries)} entries, "
          f"solvation={cfg.solvation_environment}) -> {pickle_path}")
    species_filter(
        dataset_entries=dataset_entries,
        mol_entries_pickle_location=str(pickle_path),
        species_report=species_report_path,
        species_decision_tree=tree,
        coordimer_weight=lambda mol: (mol.penalty, mol.solvation_free_energy),
        generate_unfiltered_mol_pictures=False,
    )


# ---------------------------------------------------------------------------
# Seed resolution helpers
# ---------------------------------------------------------------------------

def _lowest_G(mol_entries, formula: str, charge: int):
    """Return the index of the lowest-G coordimer matching (formula, charge),
    or None if no match. Formula whitespace is ignored ('C3H4O3' == 'C3 H4 O3')."""
    f = formula.replace(" ", "")
    cands = [e for e in mol_entries
             if e.formula.replace(" ", "") == f and e.charge == charge]
    if not cands:
        return None
    return int(min(cands, key=lambda e: e.solvation_free_energy).ind)


def _resolve_seeds_from_query(cfg: SeededConfig) -> None:
    """If cfg.seed_species_query is set, load the pickle and resolve each
    {formula, charge, count} entry to the lowest-G coordimer's index in
    the pickle. Populates cfg.seed_species and cfg.initial_state_counts
    in place.
    """
    if not cfg.seed_species_query:
        return
    with open(cfg.mol_entries_pickle, "rb") as f:
        mol_entries = pickle.load(f)

    resolved: List[int] = []
    counts: Dict[int, int] = {}
    print(f"[forge] resolving seed_species_query against "
          f"{cfg.mol_entries_pickle} ({len(mol_entries)} entries)")
    for spec in cfg.seed_species_query:
        try:
            formula = spec["formula"]; charge = int(spec["charge"])
            count = int(spec["count"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"Bad seed_species_query entry {spec!r}: each entry must "
                f"have 'formula', 'charge', and 'count'. ({e})"
            )
        idx = _lowest_G(mol_entries, formula, charge)
        if idx is None:
            raise RuntimeError(
                f"seed_species_query: no entry in pickle matches "
                f"formula={formula!r} charge={charge:+d}."
            )
        resolved.append(idx); counts[idx] = count
        e = mol_entries[idx]
        print(f"  {formula:<10} q={charge:+d}  ind={idx:<5}  "
              f"G={e.solvation_free_energy:+.3f} eV  count={count}")
    cfg.seed_species = resolved
    cfg.initial_state_counts = counts


def _resolve_seeds_from_xyz(cfg: SeededConfig) -> None:
    """If cfg.seed_species_xyz is set, resolve indices at runtime by matching
    geometry + charge + spin against the pickle.  Overwrites cfg.seed_species
    and cfg.initial_state_counts in place.
    """
    if not cfg.seed_species_xyz:
        return
    from chemistry_lib.initial_state import find_mol_entry_from_xyz_and_charge

    with open(cfg.mol_entries_pickle, "rb") as f:
        mol_entries = pickle.load(f)

    resolved: List[int] = []
    counts: Dict[int, int] = {}
    for rec in cfg.seed_species_xyz:
        xyz = rec["xyz"]; charge = int(rec["charge"])
        spin = int(rec.get("spin", 1))
        count = int(rec.get("count", 0))
        idx = find_mol_entry_from_xyz_and_charge(
            mol_entries, xyz, charge, spin_multiplicity=spin,
        )
        if idx is None:
            raise RuntimeError(
                f"Could not resolve seed species from {xyz} (charge={charge}, "
                f"spin={spin}) against {cfg.mol_entries_pickle}."
            )
        resolved.append(int(idx))
        if count > 0:
            counts[int(idx)] = count
    print(f"[forge] resolved seed_species from XYZ -> indices {resolved}")
    cfg.seed_species = resolved
    if counts:
        cfg.initial_state_counts = counts


def _validate_seeds(cfg: SeededConfig) -> None:
    """After all auto-resolution passes, the run must have at least one
    seed species AND a positive initial count for every seed."""
    if not cfg.seed_species:
        raise ValueError(
            "No seed species specified. Set one of: seed_species (with "
            "initial_state_counts), seed_species_query, or seed_species_xyz."
        )
    missing = [i for i in cfg.seed_species
               if i not in cfg.initial_state_counts
               or cfg.initial_state_counts[i] <= 0]
    if missing:
        raise ValueError(
            f"Seed species {missing} have no positive initial_state_counts "
            f"entry. Provide counts for every seed."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="CRN building with FORGE")
    ap.add_argument("--config", required=True, help="JSON config file")
    args = ap.parse_args()
    cfg = _load_config(args.config)

    _maybe_run_species_filter(cfg)
    _resolve_seeds_from_query(cfg)
    _resolve_seeds_from_xyz(cfg)
    _validate_seeds(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
