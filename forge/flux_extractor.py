"""
Extract per-species flux from a completed kMC run.

Both GMC and our Python Gillespie SSA write trajectories to initial_state.sqlite
with the schema
    trajectories(seed INTEGER, step INTEGER, reaction_id INTEGER, time REAL)
and the reaction network lives in rn.sqlite with
    reactions(reaction_id, number_of_reactants, number_of_products,
              reactant_1, reactant_2, product_1, product_2, rate, dG, ...)

We count how often each species appeared as a reactant (consumption) or a
product (production) across all firings, summed over seeds. The returned
dict is a pandas-style row per species with enough info to choose promotion.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple


def read_reactions_table(rn_sqlite: str) -> Dict[int, Tuple[List[int], List[int], float, float]]:
    """reaction_id -> (reactants, products, rate, dG)"""
    con = sqlite3.connect(rn_sqlite)
    cur = con.cursor()
    out = {}
    for row in cur.execute(
        "select reaction_id, number_of_reactants, number_of_products,"
        " reactant_1, reactant_2, product_1, product_2, rate, dG from reactions"
    ):
        rid, nr, npr, r1, r2, p1, p2, rate, dG = row
        rcts = [r for r in (r1, r2)[:nr] if r is not None and r >= 0]
        pros = [p for p in (p1, p2)[:npr] if p is not None and p >= 0]
        out[rid] = (rcts, pros, float(rate or 0.0), float(dG or 0.0))
    con.close()
    return out


def reaction_firing_counts(
    initial_state_sqlite: str,
    seeds: Optional[Iterable[int]] = None,
) -> Dict[int, int]:
    """reaction_id -> total firings across all requested seeds."""
    con = sqlite3.connect(initial_state_sqlite)
    cur = con.cursor()
    if seeds is None:
        rows = cur.execute(
            "select reaction_id, count(*) from trajectories group by reaction_id"
        ).fetchall()
    else:
        seeds = list(seeds)
        placeholder = ",".join("?" * len(seeds))
        rows = cur.execute(
            f"select reaction_id, count(*) from trajectories "
            f"where seed in ({placeholder}) group by reaction_id",
            seeds,
        ).fetchall()
    con.close()
    # defensive: drop sentinel seed rows (gmc uses -1 for pre-sim state)
    return {int(rid): int(c) for rid, c in rows if rid is not None and rid >= 0}


def compute_species_flux(
    rn_sqlite: str,
    initial_state_sqlite: str,
    seeds: Optional[Iterable[int]] = None,
) -> Dict[int, Dict[str, int]]:
    """species_id -> {produced, consumed, touched, n_rxns}.

    `touched` = produced + consumed (per-firing count).
    `n_rxns` = number of distinct reactions touching this species that fired.
    """
    rxns = read_reactions_table(rn_sqlite)
    fires = reaction_firing_counts(initial_state_sqlite, seeds=seeds)

    produced = defaultdict(int)
    consumed = defaultdict(int)
    rxns_touching = defaultdict(set)

    for rid, count in fires.items():
        if rid not in rxns:
            continue
        rcts, pros, _, _ = rxns[rid]
        for s in rcts:
            consumed[s] += count
            rxns_touching[s].add(rid)
        for s in pros:
            produced[s] += count
            rxns_touching[s].add(rid)

    all_sp = set(produced) | set(consumed)
    out: Dict[int, Dict[str, int]] = {}
    for s in all_sp:
        out[s] = {
            "produced": produced.get(s, 0),
            "consumed": consumed.get(s, 0),
            "touched": produced.get(s, 0) + consumed.get(s, 0),
            "n_rxns": len(rxns_touching[s]),
        }
    return out


def species_observed_in_initial_state(initial_state_sqlite: str) -> Set[int]:
    """Return set of species_ids that had count > 0 at t=0."""
    con = sqlite3.connect(initial_state_sqlite)
    cur = con.cursor()
    ids = {
        int(r[0])
        for r in cur.execute("select species_id from initial_state where count > 0")
    }
    con.close()
    return ids


def promotion_cutoff(
    flux: Dict[int, Dict[str, int]],
    metric: str = "produced",
    fraction: float = 1.0e-3,
) -> float:
    """Absolute-equivalent flux cutoff = `fraction` * total `metric` flux.

    `fraction` is the normalized flux threshold epsilon of the FORGE paper's
    promotion rule. Scale-invariant: removes n_simulations /
    seed-count / step_cutoff dependence so the same fraction promotes the same
    *selectivity* regardless of run scale. Computed once (not per species) to
    keep promote() O(n) and to let the driver log what selectivity a given
    fraction maps to.
    """
    total = sum(f.get(metric, 0) for f in flux.values())
    return fraction * total


def promote(
    flux: Dict[int, Dict[str, int]],
    existing_core: Iterable[int],
    metric: str = "produced",
    fraction: float = 1.0e-3,
) -> Tuple[List[int], float]:
    """Return (new species_ids to promote, effective absolute cutoff used).

    Promotes species whose share of total `metric` flux is >= `fraction`.

    metric: which field of `flux[s]` to threshold on.
        'touched'  = produced + consumed 
        'produced' = times appeared as product in fired reactions  (default; best
                     signal that a species is genuinely being made by the network)
        'consumed' = times appeared as reactant
    """
    cutoff = promotion_cutoff(flux, metric=metric, fraction=fraction)
    existing = set(int(s) for s in existing_core)
    out = [
        s for s, f in flux.items()
        if f.get(metric, 0) >= cutoff and s not in existing
    ]
    # sort by the chosen metric descending for stable reporting
    out.sort(key=lambda s: -flux[s].get(metric, 0))
    return out, cutoff


def summarize(flux: Dict[int, Dict[str, int]]) -> dict:
    if not flux:
        return {"n_species_touched": 0, "total_events": 0}
    return {
        "n_species_touched": len(flux),
        "total_events": sum(v["touched"] for v in flux.values()),
        "max_touched_species": max(flux, key=lambda s: flux[s]["touched"]),
        "max_touched_count": max(v["touched"] for v in flux.values()),
    }
