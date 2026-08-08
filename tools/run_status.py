#!/usr/bin/env python3
"""Single-shot status report for a FORGE run, with or without inline AAM,
in-flight or finished: consumes a run's out_dir (core_state.json, iter reports,
rn.sqlite, aam_cache.sqlite) and prints config, per-iteration progress, AAM
stats, kMC artifacts, and anomalies.

Invoke: python tools/run_status.py <out_dir> [<out_dir2> ...] [--json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _color(s, c):
    if not sys.stdout.isatty():
        return s
    return {"red":"\033[31m","green":"\033[32m","yellow":"\033[33m",
            "blue":"\033[34m","cyan":"\033[36m","gray":"\033[90m",
            "bold":"\033[1m","end":"\033[0m"}[c] + s + "\033[0m"

def _fmt_int(n):
    try: return f"{int(n):,}"
    except: return str(n)

def _fmt_dur(seconds):
    if seconds is None:
        return "-"
    s = float(seconds)
    if s < 60: return f"{s:.1f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.2f}h"

def _fmt_size(path):
    if not os.path.exists(path):
        return "-"
    n = os.path.getsize(path)
    for u in ("B","KB","MB","GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"

def _read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None

def _snapshot_sqlite(src):
    """Copy sqlite + WAL + SHM to /tmp for a consistent read on a shared filesystem."""
    if not os.path.exists(src):
        return None
    dst = f"/tmp/{os.environ.get('USER','anon')}_runstat_{abs(hash(src)) % 10**8}.db"
    try:
        shutil.copy(src, dst)
        for ext in ("-wal","-shm"):
            if os.path.exists(src + ext):
                shutil.copy(src + ext, dst + ext)
        return dst
    except Exception:
        return src   # fallback: try direct read

def _sql(db, q, params=()):
    if not db or not os.path.exists(db):
        return []
    try:
        c = sqlite3.connect(db, timeout=5)
        rows = c.execute(q, params).fetchall()
        c.close()
        return rows
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------

def collect(out_dir):
    """Read everything we can find in out_dir; return a dict of facts."""
    out_dir = os.path.abspath(out_dir)
    facts = {"out_dir": out_dir, "exists": os.path.isdir(out_dir)}
    if not facts["exists"]:
        return facts

    # --- core state (master record) ---
    state_path = os.path.join(out_dir, "core_state.json")
    state = _read_json(state_path)
    facts["core_state"] = state
    facts["core_state_mtime"] = os.path.getmtime(state_path) if os.path.exists(state_path) else None

    # --- final summary (config dump) ---
    summary = _read_json(os.path.join(out_dir, "final_summary.json"))
    facts["final_summary"] = summary

    # --- per-iter reports ---
    iter_reports = []
    for f in sorted(glob.glob(os.path.join(out_dir, "iter_*", "iter_report.json"))):
        d = _read_json(f)
        if d:
            d["__mtime"] = os.path.getmtime(f)
            iter_reports.append(d)
    facts["iter_reports"] = iter_reports
    facts["iter_dirs"] = sorted(glob.glob(os.path.join(out_dir, "iter_*")))

    # --- rn.sqlite ---
    rn = os.path.join(out_dir, "rn.sqlite")
    facts["rn_path"] = rn
    facts["rn_size"] = _fmt_size(rn)
    if os.path.exists(rn):
        rs = _sql(rn, """
            SELECT count(*),
              sum(CASE WHEN number_of_reactants=1 THEN 1 ELSE 0 END),
              sum(CASE WHEN number_of_reactants=2 THEN 1 ELSE 0 END)
            FROM reactions
        """)
        if rs and rs[0][0] is not None:
            facts["rn_total"] = int(rs[0][0] or 0)
            facts["rn_uni"]   = int(rs[0][1] or 0)
            facts["rn_bi"]    = int(rs[0][2] or 0)
        sp = _sql(rn, """
            SELECT count(DISTINCT id) FROM (
              SELECT reactant_1 AS id FROM reactions WHERE reactant_1>=0
              UNION SELECT reactant_2 FROM reactions WHERE reactant_2>=0
              UNION SELECT product_1  FROM reactions WHERE product_1 >=0
              UNION SELECT product_2  FROM reactions WHERE product_2 >=0)
        """)
        if sp:
            facts["rn_distinct_species"] = int(sp[0][0] or 0)
        # rate-class breakdown
        ratec = _sql(rn, """
            SELECT number_of_reactants||'->'||number_of_products, count(*)
            FROM reactions GROUP BY 1 ORDER BY 2 DESC
        """)
        facts["rn_by_rcount"] = {k: int(v) for k, v in ratec}

    # --- AAM cache ---
    cache_persistent = os.path.join(out_dir, "aam_cache.sqlite")
    facts["aam_cache_persistent_path"] = cache_persistent
    facts["aam_cache_persistent_size"] = _fmt_size(cache_persistent)
    if os.path.exists(cache_persistent):
        snap = _snapshot_sqlite(cache_persistent)
        rows = _sql(snap, "SELECT status, count(*) FROM aam_cache GROUP BY status")
        facts["aam_by_status"] = {k: int(v) for k, v in rows} if rows else {}
        agg = _sql(snap, """
            SELECT count(*),
              sum(CASE WHEN allowed=1 THEN 1 ELSE 0 END),
              avg(elapsed),
              max(elapsed),
              min(ts), max(ts)
            FROM aam_cache WHERE elapsed > 0.01
        """)
        if agg and agg[0][0] is not None:
            facts["aam_n_real"]   = int(agg[0][0] or 0)
            facts["aam_n_allowed"]= int(agg[0][1] or 0)
            facts["aam_avg_elap"] = float(agg[0][2] or 0)
            facts["aam_max_elap"] = float(agg[0][3] or 0)
            facts["aam_first_ts"] = agg[0][4]
            facts["aam_last_ts"]  = agg[0][5]
        # percentiles via Python (sqlite has no native percentile)
        elap = _sql(snap, "SELECT elapsed FROM aam_cache WHERE elapsed>0.01 ORDER BY elapsed")
        if elap:
            es = [r[0] for r in elap]
            n = len(es)
            facts["aam_p50"] = es[n//2]
            facts["aam_p95"] = es[min(int(n*0.95), n-1)]
            facts["aam_p99"] = es[min(int(n*0.99), n-1)]

        rows_total = _sql(snap, "SELECT count(*) FROM aam_cache")
        facts["aam_total"] = int(rows_total[0][0] or 0) if rows_total else 0

    # --- initial state for final kMC ---
    init = os.path.join(out_dir, "final_initial_state.sqlite")
    facts["final_init_present"] = os.path.exists(init)
    facts["final_init_size"]    = _fmt_size(init)

    # --- sink report / final kMC outputs ---
    facts["sink_report_pdf"] = os.path.exists(os.path.join(out_dir, "sink_report.pdf"))
    facts["sink_report_tex"] = os.path.exists(os.path.join(out_dir, "sink_report.tex"))

    # --- look for kMC trajectory output (varies by GMC version) ---
    facts["traj_files"] = sorted(
        glob.glob(os.path.join(out_dir, "*.traj")) +
        glob.glob(os.path.join(out_dir, "trajectories.sqlite")) +
        glob.glob(os.path.join(out_dir, "*trajectory*"))
    )

    return facts


# ---------------------------------------------------------------------------
# Variant detection
# ---------------------------------------------------------------------------

def detect_variant(facts):
    """Infer which FORGE variant produced this run."""
    cfg = (facts.get("final_summary") or {}).get("config", {})
    has_aam = bool(facts.get("aam_total"))

    if cfg.get("aam_filter"):
        return "FORGE (with AAM)"
    if has_aam:
        return "AAM cache present but aam_filter=False (mixed?)"
    return "FORGE (no AAM)"


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render(facts):
    out = []
    add = out.append

    add("=" * 92)
    add(_color("RUN STATUS: ", "bold") + facts["out_dir"])
    add("=" * 92)
    if not facts.get("exists"):
        add(_color("  !! out_dir does not exist", "red"))
        return "\n".join(out)

    # ---------------- variant ----------------
    add(_color("\nVARIANT", "cyan"))
    add(f"  detected: {detect_variant(facts)}")
    if facts.get("core_state_mtime"):
        add(f"  last activity: {datetime.fromtimestamp(facts['core_state_mtime']):%Y-%m-%d %H:%M:%S}")

    # ---------------- config ----------------
    cfg = (facts.get("final_summary") or {}).get("config", {})
    if not cfg and facts.get("iter_reports"):
        cfg = facts["iter_reports"][-1].get("gen_stats", {})  # weak fallback
    add(_color("\nCONFIG", "cyan"))
    if cfg:
        keys = [
            ("seed_species","seed species"),
            ("initial_state_counts","initial counts"),
            ("electron_free_energy","electron free energy (eV)"),
            ("temperature","temperature (K)"),
            ("max_iterations","max iterations"),
            ("promotion_flux_fraction","promotion fraction (ε)"),
            ("promotion_flux_metric","promotion metric"),
            ("n_simulations","kMC n_simulations"),
            ("step_cutoff","kMC step cutoff"),
            ("nproc","nproc"),
            ("pool_chunksize","pool chunksize"),
            ("reaction_tree","reaction tree"),
            ("aam_filter","AAM filter"),
            ("aam_timeout_sec","  AAM timeout (s)"),
            ("prefilter_bond_delta_cutoff","  pre-AAM bond Δ cutoff"),
            ("iter_wall_timeout_sec","iter wall timeout (s)"),
        ]
        for k, label in keys:
            if k in cfg and cfg[k] is not None:
                add(f"  {label:<35} {cfg[k]}")
    else:
        add("  (no final_summary.json; run may not have completed)")

    # ---------------- iterations ----------------
    add(_color("\nITERATIONS", "cyan"))
    state = facts.get("core_state") or {}
    if state:
        add(f"  last_completed_iter:  {state.get('iteration', '?')}")
        add(f"  current core size:    {len(state.get('core_ids_sorted') or state.get('core_ids') or [])}")
        promoted = state.get("per_iter_promoted") or []
        if promoted:
            add(f"  promoted per iter:    "
                + str([len(x) for x in promoted])
                + f"  (sum = {sum(len(x) for x in promoted)})")
        rxn_per_iter = state.get("per_iter_rxn_counts") or []
        if rxn_per_iter:
            add(f"  rxn added per iter:   {rxn_per_iter}")
            add(f"  total rxns from iters:{sum(int(x) for x in rxn_per_iter)}")

    iter_reports = facts.get("iter_reports") or []
    if iter_reports:
        add("")
        add(f"  {'iter':>4} {'core_in':>7} {'core_out':>8} {'rxn_new':>8} "
            f"{'tested':>11} {'aam_rej':>8} {'aam_recall':>10} "
            f"{'promoted':>8} {'wall':>8} {'finalized':>9}")
        for d in iter_reports:
            g = d.get("gen_stats") or {}
            n_rej = g.get("n_aam_rejected", 0) or 0
            n_kept = (g.get("n_kept") or g.get("n_reactions") or 0)
            recall = ""
            if n_rej + n_kept > 0:
                recall = f"{100*n_kept/(n_kept+n_rej):.0f}%"
            ef = "yes" if g.get("early_finalized") else ""
            add(f"  {d['iteration']:>4} "
                f"{d.get('core_size_in','?'):>7} "
                f"{d.get('core_size_out','?'):>8} "
                f"{_fmt_int(d.get('n_reactions_new_this_iter',0)):>8} "
                f"{_fmt_int(g.get('n_tested',0)):>11} "
                f"{_fmt_int(n_rej):>8} "
                f"{recall:>10} "
                f"{d.get('n_promoted',0):>8} "
                f"{_fmt_dur(g.get('elapsed_sec',0)):>8} "
                f"{ef:>9}")
        total_wall = sum((d.get("gen_stats") or {}).get("elapsed_sec", 0) or 0
                         for d in iter_reports)
        add(f"  {'sum':>4} {'':>7} {'':>8} "
            f"{_fmt_int(sum(d.get('n_reactions_new_this_iter',0) or 0 for d in iter_reports)):>8} "
            f"{'':>11} "
            f"{_fmt_int(sum((d.get('gen_stats') or {}).get('n_aam_rejected', 0) or 0 for d in iter_reports)):>8} "
            f"{'':>10} "
            f"{sum(d.get('n_promoted',0) or 0 for d in iter_reports):>8} "
            f"{_fmt_dur(total_wall):>8}")

    # ---------------- rn.sqlite ----------------
    add(_color("\nrn.sqlite (cumulative reaction network)", "cyan"))
    if facts.get("rn_total") is not None:
        add(f"  path:                 {facts['rn_path']}")
        add(f"  size:                 {facts['rn_size']}")
        add(f"  total reactions:      {_fmt_int(facts['rn_total'])}")
        add(f"    unimolecular:       {_fmt_int(facts.get('rn_uni',0))}")
        add(f"    bimolecular:        {_fmt_int(facts.get('rn_bi',0))}")
        add(f"  distinct species:     {_fmt_int(facts.get('rn_distinct_species',0))}")
        if facts.get("rn_by_rcount"):
            add(f"  by reactant->product class:")
            for k, v in facts["rn_by_rcount"].items():
                add(f"    {k:<8} {_fmt_int(v)}")
    else:
        add("  (rn.sqlite missing - reaction generation may not have run)")

    # ---------------- AAM stats ----------------
    if facts.get("aam_total"):
        add(_color("\nAAM CACHE STATS", "cyan"))
        add(f"  cache file:           {facts['aam_cache_persistent_path']}  "
            f"({facts['aam_cache_persistent_size']})")
        add(f"  total decisions:      {_fmt_int(facts['aam_total'])}")
        if facts.get("aam_by_status"):
            for s, c in sorted(facts["aam_by_status"].items(), key=lambda x: -x[1]):
                pct = 100*c/facts["aam_total"]
                add(f"    {s:<22} {_fmt_int(c):>8}  ({pct:5.1f}%)")
        # latency
        if "aam_avg_elap" in facts:
            add(f"  real mapper calls:    {_fmt_int(facts.get('aam_n_real',0))}")
            add(f"    mean elapsed:       {facts['aam_avg_elap']:.2f}s")
            if "aam_p50" in facts:
                add(f"    p50:                {facts['aam_p50']:.2f}s")
                add(f"    p95:                {facts['aam_p95']:.2f}s")
                add(f"    p99:                {facts['aam_p99']:.2f}s")
            add(f"    max:                {facts.get('aam_max_elap',0):.2f}s")
        # window
        if facts.get("aam_first_ts") and facts.get("aam_last_ts"):
            window = facts["aam_last_ts"] - facts["aam_first_ts"]
            add(f"  first decision at:    "
                f"{datetime.fromtimestamp(facts['aam_first_ts']):%Y-%m-%d %H:%M:%S}")
            add(f"  last  decision at:    "
                f"{datetime.fromtimestamp(facts['aam_last_ts']):%Y-%m-%d %H:%M:%S}")
            add(f"  window:               {_fmt_dur(window)}")
            if window > 0:
                add(f"  avg throughput:       {facts['aam_total']/window:.2f} dec/sec")

        # impact summary
        n_total = facts["aam_total"]
        n_allowed = facts.get("aam_by_status", {}).get("allowed", 0)
        n_rejected = facts.get("aam_by_status", {}).get("rejected", 0)
        n_failopen = (n_total - n_allowed - n_rejected)
        add("")
        add(_color("  AAM SCIENTIFIC IMPACT", "yellow"))
        add(f"    phantom reactions blocked from rn.sqlite: "
            f"{_fmt_int(n_rejected)}")
        if n_total:
            add(f"    AAM rejection rate (of validated):       "
                f"{100*n_rejected/(n_allowed+n_rejected) if (n_allowed+n_rejected) else 0:.1f}%")
        add(f"    fail-open kept (timeout/no_mapping):      "
            f"{_fmt_int(n_failopen)}")
        # sum of aam_rej from iters (cross-check)
        sum_iter_rej = sum((d.get("gen_stats") or {}).get("n_aam_rejected", 0) or 0
                           for d in iter_reports)
        if sum_iter_rej:
            add(f"    total AAM rejections summed over iters:  "
                f"{_fmt_int(sum_iter_rej)}")
    else:
        add(_color("\nAAM CACHE STATS", "cyan"))
        add("  (no aam_cache.sqlite present - this run did not use inline AAM)")

    # ---------------- final kMC artifacts ----------------
    add(_color("\nFINAL kMC ARTIFACTS", "cyan"))
    add(f"  final_initial_state.sqlite:  "
        f"{'present (' + facts['final_init_size'] + ')' if facts['final_init_present'] else '- not built -'}")
    add(f"  trajectory files:            "
        f"{', '.join(os.path.basename(p) for p in facts['traj_files']) or '- none found -'}")
    add(f"  sink_report.tex:             {'yes' if facts['sink_report_tex'] else 'no'}")
    add(f"  sink_report.pdf:             {'yes' if facts['sink_report_pdf'] else 'no'}")

    # ---------------- anomalies ----------------
    add(_color("\nANOMALIES / DIAGNOSTICS", "cyan"))
    issues = []
    # 1. rn.sqlite > expected?
    if facts.get("rn_total") is not None and state and state.get("per_iter_rxn_counts"):
        expected = sum(int(x) for x in state["per_iter_rxn_counts"])
        actual = facts["rn_total"]
        if actual > expected + 1:
            issues.append(_color(
                f"WARN: rn.sqlite has {_fmt_int(actual)} rows but per_iter_rxn_counts "
                f"sums to {_fmt_int(expected)}. Excess = {_fmt_int(actual-expected)}. "
                f"Previous run was killed mid-iter; v2 auto-trim will fix on resume.",
                "yellow"))
        elif actual < expected:
            issues.append(_color(
                f"WARN: rn.sqlite has {_fmt_int(actual)} rows but expected {_fmt_int(expected)} "
                f"from per_iter_rxn_counts. Possible corruption.", "red"))
    # 2. iter that finalized early
    early = [d['iteration'] for d in iter_reports
             if (d.get('gen_stats') or {}).get('early_finalized')]
    if early:
        issues.append(_color(
            f"INFO: iter(s) {early} hit wall timeout and finalized early "
            f"(v2 finalize-and-continue).", "yellow"))
    # 3. AAM hard timeouts present?
    n_hard = facts.get("aam_by_status", {}).get("hard_timeout", 0)
    n_soft = facts.get("aam_by_status", {}).get("timeout", 0)
    if n_hard or n_soft:
        issues.append(
            f"INFO: AAM hit timeout {n_hard+n_soft} times "
            f"(hard_kill={n_hard}, signal={n_soft}). "
            f"All kept fail-open.")
    # 4. recent core_state freshness
    if facts.get("core_state_mtime"):
        age = time.time() - facts["core_state_mtime"]
        if age > 24*3600:
            issues.append(_color(
                f"INFO: last state save was {_fmt_dur(age)} ago - run is idle/finished.",
                "gray"))
    if not issues:
        add(_color("  ok no anomalies detected", "green"))
    else:
        for x in issues:
            add(f"  {x}")

    add("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="run out_dir(s) to inspect")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of human report")
    args = ap.parse_args()

    if args.json:
        out = {d: collect(d) for d in args.dirs}
        # strip non-JSON values
        def clean(o):
            if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)): return [clean(x) for x in o]
            if isinstance(o, (int, float, str, bool)) or o is None: return o
            try:
                json.dumps(o); return o
            except Exception:
                return str(o)
        print(json.dumps(clean(out), indent=2, default=str))
        return

    for d in args.dirs:
        facts = collect(d)
        print(render(facts))


if __name__ == "__main__":
    main()
