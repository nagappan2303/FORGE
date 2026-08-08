"""
AAM filter with subprocess-isolated hard timeout.

A cheap structural pre-filter (bond-delta cap) drops obviously non-elementary
candidates before the AAM call. The mapper itself enforces the strict
elementary-step criterion (bond edits n_b + n_f <= 2 with a shared atom)
internally; see the FORGE paper's AAM accept criterion.

Indecisive mapper outcomes (timeout, no_data, errors) fall back to fail-open
KEEP (controlled by `keep_on_timeout`, default True). The cache path can be
overridden via the AAM_CACHE_PATH env var (used by SLURM scripts to point at
/tmp).
"""
from __future__ import annotations

import os
import re
import signal
import time
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_initialized = False
_analyze_reaction = None
_molgraph_to_smiles = None
_MoleculeGraph = None
_OpenBabelNN = None

_smiles_cache: Dict[int, Optional[str]] = {}
_bondcount_cache: Dict[int, int] = {}


def _formula_canon(formula_str):
    out = {}
    for tok in formula_str.split():
        m = re.match(r'([A-Za-z]+)(\d+)', tok)
        if m: out[m.group(1)] = int(m.group(2))
    return tuple(sorted(out.items()))


_MANUAL_SMILES_BY_FORMULA = {
    (_formula_canon("F6 P1"), -1): "F[P-](F)(F)(F)(F)F",
}

_manual_overrides_by_ind: Dict[int, str] = {}


def _seed_manual_overrides(mol_entries):
    if _manual_overrides_by_ind:
        return
    for e in mol_entries:
        key = (_formula_canon(e.formula), e.charge)
        if key in _MANUAL_SMILES_BY_FORMULA:
            _manual_overrides_by_ind[e.ind] = _MANUAL_SMILES_BY_FORMULA[key]


def _lazy_init():
    global _initialized, _analyze_reaction, _molgraph_to_smiles
    global _MoleculeGraph, _OpenBabelNN
    if _initialized:
        return

    # Cap BLAS thread pools at 1: the mapper is fork-fanned out at the
    # FORGE process level, so any per-call multithreading just contends.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")

    from aam_src.reaction_mapper import analyze_reaction
    from chemistry_lib.reaction_questions import molgraph_to_smiles
    from pymatgen.analysis.graphs import MoleculeGraph
    from pymatgen.analysis.local_env import OpenBabelNN

    _analyze_reaction = analyze_reaction
    _molgraph_to_smiles = molgraph_to_smiles
    _MoleculeGraph = MoleculeGraph
    _OpenBabelNN = OpenBabelNN

    # Derive the coordinate-pair set from the mapper's single source of
    # truth so the prefilter can never drift from the mapping stage again.
    global _COORD_TYPES
    try:
        import aam_src.MapIsing as _mi
        _COORD_TYPES = {tuple(sorted(fs)) for fs in _mi.COORD_PAIRS_SYM}
    except Exception:
        pass  # keep the static fallback defined at module level

    _initialized = True


# ---------------------------------------------------------------------------
# SMILES helpers
# ---------------------------------------------------------------------------

def _get_smiles(mol_entry) -> Optional[str]:
    ind = mol_entry.ind
    if ind in _smiles_cache:
        return _smiles_cache[ind]
    if ind in _manual_overrides_by_ind:
        smi = _manual_overrides_by_ind[ind]
        _smiles_cache[ind] = smi
        return smi
    try:
        # Prefer the MoleculeGraph stored on the entry at species-filter
        # time: rebuilding one from the raw molecule can fail when the
        # entries were serialized by a different pymatgen version.
        mg = getattr(mol_entry, "molecule_graph", None)
        if mg is None:
            mg = getattr(mol_entry, "mol_graph", None)
        if mg is None:
            mg = _MoleculeGraph.with_local_env_strategy(
                mol_entry.molecule, _OpenBabelNN())
        smi = _molgraph_to_smiles(mg, sanitize=True, remove_hs=False)
    except Exception:
        smi = None
    _smiles_cache[ind] = smi
    return smi


# ---------------------------------------------------------------------------
# Cheap structural pre-filter
# ---------------------------------------------------------------------------

def _bond_hist(mol_entry):
    """Per-bond-type histogram of the covalent graph: Counter keyed by the
    sorted element pair of each edge, e.g. ('C','O'), ('H','O'), ('C','F')."""
    from collections import Counter
    ind = mol_entry.ind
    h = _bondcount_cache.get(ind)
    if h is not None:
        return h
    g = getattr(mol_entry, "covalent_graph", None) or getattr(mol_entry, "graph", None)
    h = Counter()
    try:
        if g is not None:
            syms = [str(getattr(s, "specie", None).symbol) if getattr(s, "specie", None) is not None
            else str(s.species.elements[0].symbol) for s in mol_entry.molecule]
            for u, v, *_rest in g.edges():
                h[tuple(sorted((syms[u], syms[v])))] += 1
    except Exception:
        h = Counter()
    _bondcount_cache[ind] = h
    return h


# Coordinate (dative) metal-oxygen bond types, excluded from the typed delta
# so this prefilter agrees with the decision tree's bond_type_change_filter
# and the AAM stage (both exempt O-Li and O-Na). Extend HERE and in
# MapIsing.COORD_PAIRS_AN / COORD_PAIRS_SYM together, never one alone.
_COORD_TYPES = {("Li", "O"), ("Na", "O")}


def cheap_prefilter(reaction: dict, mol_entries,
                    max_bond_delta: int = 3) -> Tuple[bool, str]:
    """Returns (passes, reason). passes=False means DROP this reaction
    pre-AAM. Metric: summed absolute per-bond-type count difference between
    reactants and products (coordinate metal-oxygen types excluded, matching
    the AAM stage). An elementary step is bounded by 2 (break-1 and
    form-1 of different types); the default cutoff 3 adds a margin for
    bond-perception mismatches between the connectivity graphs used here and
    the kekulized representation used by the mapper. With a decision tree
    that includes fragment matching upstream, the bound is already
    guaranteed and this filter acts as an O(1) safety net. The mapper itself
    enforces the strict elementary criterion (n_broken + n_formed <= 2 with
    a shared reaction center) internally."""
    try:
        from collections import Counter
        hr, hp = Counter(), Counter()
        for i in reaction["reactants"]:
            hr += _bond_hist(mol_entries[i])
        for i in reaction["products"]:
            hp += _bond_hist(mol_entries[i])
        delta = sum(abs(hp.get(k, 0) - hr.get(k, 0))
                    for k in set(hr) | set(hp) if k not in _COORD_TYPES)
    except Exception:
        return (True, "")
    if delta > max_bond_delta:
        return (False, f"bond_delta_{delta}")
    return (True, "")


# ---------------------------------------------------------------------------
# Hard subprocess timeout for AAM call
# ---------------------------------------------------------------------------

def _analyze_reaction_hardkill(rxn_str: str, timeout_s: int) -> dict:
    """
    Run analyze_reaction in a forked subprocess. SIGKILL on hard timeout.
    SIGKILL is OS-level and unblockable: the child IS dead.

    Returns the analyze_reaction dict, or a synthetic dict with
    map_status='hard_timeout' / 'subprocess_error' / 'no_data' / 'parse_error'.
    """
    import json, select

    rd, wr = os.pipe()
    pid = os.fork()

    if pid == 0:
        # ----- child -----
        os.close(rd)
        try:
            info = _analyze_reaction(rxn_str, timeout_s=timeout_s)
            payload = json.dumps(_jsonable(info))
        except Exception as e:
            payload = json.dumps({
                "map_status": "child_error",
                "msg": f"{type(e).__name__}: {e}",
                "allowed_reaction": False,
            })
        try:
            os.write(wr, payload.encode("utf-8"))
        except Exception:
            pass
        finally:
            try: os.close(wr)
            except Exception: pass
            os._exit(0)

    # ----- parent -----
    os.close(wr)
    deadline = time.time() + timeout_s + 5  # 5s grace for clean shutdown
    data = b""
    try:
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                # HARD TIMEOUT: SIGKILL the child
                try: os.kill(pid, signal.SIGKILL)
                except ProcessLookupError: pass
                try: os.waitpid(pid, 0)
                except ChildProcessError: pass
                return {"map_status": "hard_timeout",
                        "allowed_reaction": True}
            rlist, _, _ = select.select([rd], [], [], remaining)
            if rd not in rlist:
                continue
            chunk = os.read(rd, 65536)
            if not chunk:
                break
            data += chunk
    finally:
        try: os.close(rd)
        except Exception: pass
        try: os.waitpid(pid, 0)
        except ChildProcessError: pass

    if not data:
        return {"map_status": "no_data", "allowed_reaction": True}
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        return {"map_status": f"parse_error: {e}",
                "allowed_reaction": True}


def _jsonable(o):
    """Make analyze_reaction's return value JSON-serializable."""
    import json
    try:
        json.dumps(o)
        return o
    except (TypeError, ValueError):
        if isinstance(o, dict):
            return {str(k): _jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_jsonable(x) for x in o]
        return str(o)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aam_check_reaction(
    reaction: dict,
    mol_entries,
    timeout_s: int = 60,
    keep_on_timeout: bool = True,
    cache_path: Optional[str] = None,
    pre_filter: bool = True,
    max_bond_delta: int = 3,
) -> Tuple[bool, dict]:
    """
    Decide whether a reaction passes the AAM elementary-step filter.

    Pipeline:
      1. cheap_prefilter (bond-count delta): drops obviously non-elementary
      2. SMILES build for both sides
      3. cache lookup (canonical reaction SMILES key)
      4. analyze_reaction in a forked subprocess with SIGKILL timeout
      5. cache write-through

    Indecisive mapper outcomes (timeout, no_data, errors) fall back to
    fail-open: the reaction is KEPT (decision = `keep_on_timeout`, default
    True) so that pathological mapper cases don't silently delete chemistry.

    Returns (allowed, info-dict).
    """
    _lazy_init()
    _seed_manual_overrides(mol_entries)

    # Explicit argument wins; the env var is only a fallback when no path
    # was routed, so a variable leaked into the environment can never
    # override an explicitly-routed cache path (two-tier local, per-rank,
    # and batch alike).
    if cache_path is None:
        cache_path = os.environ.get("AAM_CACHE_PATH") or None

    # 1. cheap pre-filter (bond delta)
    if pre_filter:
        passes, reason = cheap_prefilter(reaction, mol_entries,
                                         max_bond_delta=max_bond_delta)
        if not passes:
            return (False, {"map_status": f"prefilter_drop:{reason}"})

    # 2. SMILES build: failures are FAIL-OPEN KEEPS (the mapper never got to
    # render a verdict; a graph-perception failure must not silently delete
    # every reaction touching an unresolvable species).
    try:
        reactants_smi = []
        for ind in reaction["reactants"]:
            smi = _get_smiles(mol_entries[ind])
            if not smi:  # None or empty string; both would corrupt rxn_str
                return (bool(keep_on_timeout),
                        {"map_status": "bad_smiles_reactant", "ind": ind})
            reactants_smi.append(smi)
        products_smi = []
        for ind in reaction["products"]:
            smi = _get_smiles(mol_entries[ind])
            if not smi:  # None or empty string; both would corrupt rxn_str
                return (bool(keep_on_timeout),
                        {"map_status": "bad_smiles_product", "ind": ind})
            products_smi.append(smi)
    except Exception as e:
        return (bool(keep_on_timeout),
                {"map_status": f"smiles_error: {type(e).__name__}: {e}"})

    # 3. cache lookup
    key = None
    if cache_path:
        from forge.aam_cache import canonical_key, lookup
        key = canonical_key(reactants_smi, products_smi)
        hit = lookup(cache_path, key)
        if hit is not None:
            allowed, status = hit
            if status == "no_mapping":
                # no_mapping is a deterministic mapper OUTCOME but its keep
                # decision is run POLICY: re-derive from the CURRENT
                # keep_on_timeout instead of replaying the caching run's.
                allowed = bool(keep_on_timeout)
            return (allowed, {"map_status": f"cache:{status}", "cache_hit": True})

    # 4. mapper (hard-killed subprocess; uniform timeout for all reactions)
    rxn_str = ".".join(reactants_smi) + ">>" + ".".join(products_smi)
    t0 = time.time()
    info = _analyze_reaction_hardkill(rxn_str, timeout_s)
    elapsed = time.time() - t0

    allowed = bool(info.get("allowed_reaction", False))
    status = info.get("map_status", "")

    # ALLOW-LIST GATE: only a decisive mapper verdict (map_status == 'mapped')
    # may reject a reaction. Every other status is an AAM *failure*, not a
    # verdict, and fails open via keep_on_timeout: timeout, no_mapping,
    # subprocess/parse errors, AND mapper-side SMILES re-parse failures
    # (bad_reaction_string / empty_side / bad_reactant_smiles /
    # bad_product_smiles, which arrive with allowed_reaction=False).
    # store() persists only deterministic statuses (allowed/rejected/
    # no_mapping), so failure outcomes are never frozen into the cache.
    if status != "mapped":
        decision = bool(keep_on_timeout)
        if cache_path and key is not None:
            from forge.aam_cache import store
            store(cache_path, key, decision, status or "unknown", elapsed)
        return (decision, info)

    # 5. cache final decision (decisive 'mapped' verdict only)
    if cache_path and key is not None:
        from forge.aam_cache import store
        store(cache_path, key, allowed,
              "allowed" if allowed else "rejected", elapsed)
    return (allowed, info)
