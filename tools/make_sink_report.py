#!/usr/bin/env python3
"""Rebuild sink_report and species_report for a finished FORGE run. Consumes the out_dir's
rn.sqlite, final_kmc/ trajectories, and mol_entries pickle; writes .tex reports
and PDFs (via pdflatex) under final_kmc/.

Invoke: python tools/make_sink_report.py <out_dir> [<out_dir2> ...]
        [--mol-pickle PATH] [--skip-pdflatex]
"""
from __future__ import annotations
import argparse, json, os, subprocess, traceback
from pathlib import Path

from chemistry_lib.network_loader import NetworkLoader
from chemistry_lib.report_generator import ReportGenerator
from chemistry_lib.mc_analysis import (
    sink_report,
    SimulationReplayer,
    species_report,
)

def find_mol_pickle(out_dir, override=None):
    if override and os.path.exists(override):
        return override
    for name in ("config_used.json", "final_summary.json"):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            d = json.loads(Path(p).read_text())
            if 'config' in d:
                d = d['config']
            mp = d.get('mol_entries_pickle')
            if mp and os.path.exists(mp):
                return mp
    raise FileNotFoundError(f"could not determine mol_entries_pickle for {out_dir}")


def process_one(out_dir, mol_pickle_override=None, skip_pdflatex=False):
    out_dir = os.path.abspath(out_dir)
    final_kmc = os.path.join(out_dir, "final_kmc")
    rn_db   = os.path.join(out_dir, "rn.sqlite")
    init_db = os.path.join(final_kmc, "initial_state.sqlite")

    print(f"\n{'='*78}\n>> {out_dir}\n{'='*78}")

    for p, label in [(final_kmc, "final_kmc/"), (rn_db, "rn.sqlite"),
                     (init_db, "final_kmc/initial_state.sqlite")]:
        if not os.path.exists(p):
            print(f"  !! missing: {label} ({p})")
            print(f"  skipping this run.")
            return False

    try:
        mol_pickle = find_mol_pickle(out_dir, mol_pickle_override)
    except Exception as e:
        print(f"  !! {e}")
        return False

    print(f"  rn:         {rn_db}")
    print(f"  init:       {init_db}")
    print(f"  mol pickle: {mol_pickle}")

    print("  [1/4] loading NetworkLoader...")
    nl = NetworkLoader(rn_db, mol_pickle, init_db)
    nl.load_trajectories()
    nl.load_initial_state()
    print(f"        n_species={len(nl.mol_entries)}  "
          f"n_reactions={getattr(nl, 'number_of_reactions', '?')}")

    print("  [2/4] mol_pictures (rebuilds first time; cached after)...")
    mp_dir = os.path.join(final_kmc, "mol_pictures")
    rebuild = not os.path.isdir(mp_dir) or not os.listdir(mp_dir)
    print(f"        rebuild_mol_pictures={rebuild}")
    ReportGenerator(nl.mol_entries,
                    os.path.join(final_kmc, "dummy.tex"),
                    rebuild_mol_pictures=rebuild)

    print("  [3/4] SimulationReplayer (replays kMC trajectories)...")
    sr = SimulationReplayer(nl)

    print("  [4/4] writing reports...")
    sink_tex = os.path.join(final_kmc, "sink_report.tex")
    species_tex = os.path.join(final_kmc, "species_report.tex")
    try:
        sink_report(sr, sink_tex)
        print(f"        ok {sink_tex}")
    except Exception:
        print(f"        FAIL sink_report failed:")
        traceback.print_exc()
    try:
        species_report(nl, species_tex)
        print(f"        ok {species_tex}")
    except Exception:
        print(f"        FAIL species_report failed:")
        traceback.print_exc()

    if skip_pdflatex:
        print("  (skipping pdflatex)")
        return True

    print("  pdflatex (2 passes)...")
    cwd = os.getcwd()
    try:
        os.chdir(final_kmc)
        for tex_name in ("sink_report.tex", "species_report.tex"):
            if not os.path.exists(tex_name): continue
            for _ in range(2):
                subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_name],
                               capture_output=True, timeout=300)
            pdf_name = tex_name.replace(".tex", ".pdf")
            mark = "ok" if os.path.exists(pdf_name) else "FAIL"
            print(f"        {mark} {pdf_name}")
    finally:
        os.chdir(cwd)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="run out_dirs to process")
    ap.add_argument("--mol-pickle", default=None,
                    help="explicit mol_entries.pickle path (overrides config)")
    ap.add_argument("--skip-pdflatex", action="store_true")
    args = ap.parse_args()

    n_ok = 0
    for d in args.dirs:
        try:
            if process_one(d, args.mol_pickle, args.skip_pdflatex):
                n_ok += 1
        except Exception:
            print(f"  FAIL unexpected exception:")
            traceback.print_exc()
    print(f"\n{n_ok}/{len(args.dirs)} runs processed successfully.")


if __name__ == "__main__":
    main()
