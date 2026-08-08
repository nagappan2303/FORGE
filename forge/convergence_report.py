"""
Read an out_dir produced by iteration_driver.run() and print a compact
convergence report. Also produces a markdown summary.

Usage:
    python -m forge.convergence_report --out-dir ./small_test_out
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def report(out_dir: str) -> dict:
    out = Path(out_dir)
    core = json.loads((out / "core_state.json").read_text())
    final = None
    fs = out / "final_summary.json"
    if fs.exists():
        final = json.loads(fs.read_text())

    iter_reports = []
    for d in sorted(out.glob("iter_*")):
        rep = d / "iter_report.json"
        if rep.exists():
            iter_reports.append(json.loads(rep.read_text()))

    # mol entries for pretty names
    mep = out / "mol_entries.pickle"
    id2info: dict = {}
    if mep.exists():
        with open(mep, "rb") as f:
            me = pickle.load(f)
        for e in me:
            id2info[int(e.ind)] = {
                "entry_id": e.entry_id,
                "formula": e.formula,
                "charge": e.charge,
                "spin": e.spin_multiplicity,
                "natoms": e.num_atoms,
                "G": getattr(e, "solvation_free_energy", None),
            }

    # ---- console print ----
    print(f"\n==== FORGE convergence report: {out_dir} ====")
    print(f"iterations run   : {len(iter_reports)}")
    print(f"final core size  : {core['core_size']}")
    print(f"seed + promoted  : {[len(x) for x in core['per_iter_promoted']]}")
    print(f"rxns per iter    : {core['per_iter_rxn_counts']}")
    print()
    print(f"{'iter':<6}{'core_in':<10}{'core_out':<10}{'n_pair_buckets':<16}"
          f"{'n_rxns':<10}{'n_touched':<12}{'n_promoted':<12}")
    for r in iter_reports:
        print(f"  {r['iteration']:<4}{r['core_size_in']:<10}"
              f"{r['core_size_out']:<10}{r['bucket_stats']['n_pair']:<16}"
              f"{r.get('n_reactions_new_this_iter', r.get('n_reactions', 0)):<10}"
              f"{r['flux_summary'].get('n_species_touched',0):<12}"
              f"{r['n_promoted']:<12}")

    # ---- core composition ----
    print("\n==== final core species ====")
    for sid in core["core_ids_sorted"]:
        info = id2info.get(int(sid), {})
        promoted_at = core.get("promoted_at", {}).get(str(sid))
        if promoted_at is None:
            promoted_at = core.get("promoted_at", {}).get(int(sid))
        entry_id = info.get("entry_id", "?")
        formula = info.get("formula", "?")
        charge = info.get("charge", "?")
        spin = info.get("spin", "?")
        G = info.get("G")
        G_str = f"{G:.2f}" if isinstance(G, float) else str(G)
        tag = f"(seed)" if promoted_at == 0 else f"(iter {promoted_at})"
        print(f"  ind={sid:>4} {entry_id:>15}  {formula:<20} q={charge:+d} s={spin}  G={G_str:<12} {tag}")

    # ---- markdown summary ----
    md = []
    md.append(f"# FORGE run summary")
    md.append(f"- out_dir: `{out_dir}`")
    md.append(f"- iterations run: {len(iter_reports)}")
    md.append(f"- final core size: {core['core_size']}")
    md.append(f"- reactions per iter: {core['per_iter_rxn_counts']}")
    md.append(f"- cumulative reactions: {sum(core['per_iter_rxn_counts'])}")
    md.append("")
    md.append("## Per-iteration stats")
    md.append("| iter | core_in | core_out | pair_buckets | reactions | species_touched | promoted |")
    md.append("|---|---|---|---|---|---|---|")
    for r in iter_reports:
        md.append(f"| {r['iteration']} | {r['core_size_in']} | {r['core_size_out']} | "
                  f"{r['bucket_stats']['n_pair']} | "
                  f"{r.get('n_reactions_new_this_iter', r.get('n_reactions', 0))} | "
                  f"{r['flux_summary'].get('n_species_touched',0)} | {r['n_promoted']} |")
    md.append("")
    md.append("## Final core species")
    md.append("| ind | entry_id | formula | charge | spin | G (eV) | first promoted |")
    md.append("|---|---|---|---|---|---|---|")
    for sid in core["core_ids_sorted"]:
        info = id2info.get(int(sid), {})
        pa = core.get("promoted_at", {}).get(str(sid)) or core.get("promoted_at", {}).get(int(sid)) or 0
        G = info.get("G")
        G_str = f"{G:.2f}" if isinstance(G, float) else "?"
        md.append(f"| {sid} | {info.get('entry_id','?')} | {info.get('formula','?')} | "
                  f"{info.get('charge','?'):+d} | {info.get('spin','?')} | {G_str} | "
                  f"{'seed' if pa == 0 else f'iter {pa}'} |")

    md_path = out / "convergence_report.md"
    md_path.write_text("\n".join(md))
    print(f"\nWrote {md_path}")

    # ---- optional plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        iters = [r["iteration"] for r in iter_reports]
        rxns = [r.get("n_reactions_new_this_iter", r.get("n_reactions", 0))
                for r in iter_reports]
        promoted = [r["n_promoted"] for r in iter_reports]
        core_out = [r["core_size_out"] for r in iter_reports]

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].plot(iters, core_out, "o-", label="core size (cum)")
        axes[0].set_xlabel("iteration")
        axes[0].set_ylabel("species")
        axes[0].set_title("Core size growth")
        axes[0].grid(alpha=0.3)

        axes[1].bar(iters, rxns, label="reactions/iter")
        axes[1].set_xlabel("iteration")
        axes[1].set_ylabel("reactions generated")
        axes[1].set_title("Reactions per iteration")
        axes[1].grid(alpha=0.3, axis="y")

        axes[2].bar(iters, promoted, color="C2", label="promoted/iter")
        axes[2].set_xlabel("iteration")
        axes[2].set_ylabel("new species")
        axes[2].set_title("Promotions per iteration")
        axes[2].grid(alpha=0.3, axis="y")

        fig.tight_layout()
        png_path = out / "convergence_report.png"
        fig.savefig(png_path, dpi=140)
        print(f"Wrote {png_path}")
    except Exception as ex:
        print(f"(plotting skipped: {ex})")

    return {
        "n_iterations": len(iter_reports),
        "final_core": core["core_size"],
        "cum_reactions": sum(core["per_iter_rxn_counts"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    report(args.out_dir)


if __name__ == "__main__":
    main()
