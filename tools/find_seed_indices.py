"""Look up seed species indices by formula:charge query: consumes a species
JSON (running species_filter to build mol_entries.pickle if missing) and
prints each match's index and free energy for a config's `seed_species` and
`initial_state_counts`.

Invoke: python tools/find_seed_indices.py --species-json J.json --pickle-out P \\
        --solvation li_ec --out-dir OUT --query Li1:+1 C3H4O3:0 F4O1P1:-1
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from forge.config import SeededConfig
from forge.run_seeded import _maybe_run_species_filter


def lowest_G(entries, formula: str, charge: int):
    f = formula.replace(" ", "")
    cands = [
        e for e in entries
        if e.formula.replace(" ", "") == f and e.charge == charge
    ]
    if not cands:
        return None
    return min(cands, key=lambda e: e.solvation_free_energy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species-json", required=True)
    ap.add_argument("--pickle-out", required=True,
                    help="path where the pickle should land (under out_dir)")
    ap.add_argument("--solvation", required=True,
                    help="'li_ec' or 'Na_ec'")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--query", nargs="+", required=True,
        help='Each spec is FORMULA:CHARGE, e.g. "Li1:+1" "C3H4O3:0" "F4O1P1:-1"',
    )
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    cfg = SeededConfig(
        mol_entries_pickle=args.pickle_out,
        out_dir=args.out_dir,
        electron_free_energy=0.0,           # not used by species_filter
        species_json_path=args.species_json,
        solvation_environment=args.solvation,
    )
    _maybe_run_species_filter(cfg)

    with open(args.pickle_out, "rb") as f:
        entries = pickle.load(f)

    print(f"\n=== seed-species lookup ({len(entries)} entries in pickle) ===")
    for spec in args.query:
        formula, charge_s = spec.split(":")
        charge = int(charge_s)
        e = lowest_G(entries, formula, charge)
        if e is None:
            print(f"  {formula:<10} q={charge:+d}  NOT FOUND")
        else:
            print(f"  {formula:<10} q={charge:+d}  ind={e.ind:<5}  "
                  f"G={e.solvation_free_energy:+.3f} eV  entry_id={e.entry_id}")


if __name__ == "__main__":
    main()
