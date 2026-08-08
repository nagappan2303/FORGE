#!/usr/bin/env python
import os
import time
import signal
import resource
from collections import Counter
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*') 

# ---- import your mapper stack ----
import MapIsing as mi
import networkx as nx
import optim_wrapper as ow


class _MapTimeout(Exception):
    pass


def _alarm_handler(sig, frame):
    raise _MapTimeout


def _tag(idx, sym_map):
    """'4' + {4:'O'} -> '4(O)'.  Falls back to bare index if missing."""
    s = sym_map.get(idx) if sym_map else None
    return f"{idx}({s})" if s else str(idx)


def serialize_bond_list(bond_list, atom_symbols=None):
    """
    Convert iterable of tuples like [(1,2), (3,4), (1,2)] ->
        '1(C)-2(O);1(C)-2(O);3(C)-4(O)'   (when atom_symbols is given)
        '1-2;1-2;3-4'                      (otherwise)

    Duplicates are preserved (used to represent multi-bond order).
    atom_symbols should be a dict {atom_index_1based: element_symbol}.
    """
    if not bond_list:
        return ""
    return ";".join(
        f"{_tag(i, atom_symbols)}-{_tag(j, atom_symbols)}"
        for i, j in sorted(bond_list)
    )


def _elementary_tiebreak(n_broken, n_formed, broken_list, formed_list):
    """0 if this mapping satisfies the elementary-step criterion (preferred on
    cost ties), else 1. Without this term, a disjoint (1,1) mapping that ties
    a shared (1,1) mapping at identical edit cost wins or loses purely on
    clique/permutation enumeration order, making the verdict order-dependent
    and falsely rejecting genuinely elementary reactions. The reaction is
    elementary if ANY minimal-cost mapping is an elementary step, so ties
    must resolve toward the elementary representative."""
    if n_broken == 0 and n_formed <= 1:
        return 0
    if n_broken == 1 and n_formed == 0:
        return 0
    if (n_broken == 1 and n_formed == 1
            and set(broken_list[0]) & set(formed_list[0])):
        return 0
    return 1


def serialize_mapping(mapping_tup, r_atom_symbols=None, p_atom_symbols=None):
    """
    mapping_tup is an iterable of (reactant_atom, product_atom).
    Convert to 'r1(Li):p1(Li);r2(O):p2(O);...' when symbol dicts are
    given, or 'r1:p1;r2:p2;...' otherwise.
    """
    if not mapping_tup:
        return ""
    try:
        pairs = sorted((int(r), int(p)) for r, p in dict(mapping_tup).items())
        return ";".join(
            f"{_tag(r, r_atom_symbols)}:{_tag(p, p_atom_symbols)}"
            for r, p in pairs
        )
    except Exception:
        return ""


def serialize_order_changes(order_changes, atom_symbols=None):
    """
    Convert [((i,j), r_order, p_order), ...] ->
        '1(C)-2(O):1>2;3(C)-4(O):2>1'   (when atom_symbols is given)
        '1-2:1>2;3-4:2>1'                (otherwise)
    """
    if not order_changes:
        return ""
    return ";".join(
        f"{_tag(i, atom_symbols)}-{_tag(j, atom_symbols)}:{r}>{p}"
        for ((i, j), r, p) in sorted(order_changes)
    )


def bonds_and_symbols(full_side_smiles):
    """
    Build bond MULTISET (Counter) and atom symbol dict for an entire reaction side
    (may contain '.' for multiple fragments).

    Each bond is counted by its integer order, so a double bond contributes
    the pair twice and a triple bond contributes it three times.  This way,
    breaking a C=C entirely shows up as 2 bond-units broken on the same pair.

    Processes the full SMILES as one molecule so that atom indices are
    consistent with MapIsing.

    Atom indices are 1-based, no offset.
    """
    m = Chem.MolFromSmiles(full_side_smiles)
    if m is None:
        return None, None

    Chem.Kekulize(m, clearAromaticFlags=True)   # match MapIsing treatment
    m = Chem.AddHs(m)                           # explicit H

    bonds = Counter()
    atoms = {i + 1: a.GetSymbol() for i, a in enumerate(m.GetAtoms())}

    for b in m.GetBonds():
        i = b.GetBeginAtomIdx() + 1
        j = b.GetEndAtomIdx() + 1
        order = int(b.GetBondTypeAsDouble())    # 1, 2, or 3
        if order < 1:
            order = 1                            # treat any odd type as single
        pair = (min(i, j), max(i, j))
        bonds[pair] += order

    return bonds, atoms


# ---------------------------------------------------------------------
# Trivial fast-path for reactions whose heavy-atom bond graph is empty
# on at least one side -- e.g. F.[H] >> [H][H].[F], H. + F2 >> HF + F.,
# or any reaction where every heavy atom is only bonded to hydrogens.
#
# For these cases the max-clique / modular-product machinery has nothing
# to grip onto (bond-mode requires at least one heavy-heavy bond on each
# side).  Instead of falling through to an atom-mode retry or a
# singleton-clique enumeration -- both of which can hang a worker on
# large graphs and OOM-cascade into pool_broken -- we handle the
# trivial case directly by enumerating element-preserving permutations.
#
# The permutation count is bounded by TRIVIAL_MAP_PERM_CAP; the kind of
# reactions we use this for have <= 4 heavy atoms and <= 6 H's, so the
# count is a handful, not an exponential explosion.
# ---------------------------------------------------------------------
TRIVIAL_MAP_PERM_CAP = 100_000


def _has_heavy_heavy_bond(bonds, atom_symbols):
    """True iff any bond in `bonds` connects two non-H atoms."""
    for (i, j) in bonds:
        if atom_symbols.get(i) != "H" and atom_symbols.get(j) != "H":
            return True
    return False


def _trivial_map_by_permutation(reactant_bonds, atomR,
                                product_bonds, atomP,
                                edits_for_mapping,
                                perm_cap=TRIVIAL_MAP_PERM_CAP):
    """
    Enumerate every element-preserving atom mapping between the two sides
    and return the one with the minimum edit score.  Intended only for
    trivially small reactions (no heavy-heavy bonds on at least one side)
    -- does NOT touch MapIsing / dwave-neal / networkx.

    Returns (key, tup, broken_list, formed_list, order_changes) or None.
    """
    from itertools import permutations
    from math import factorial
    from collections import defaultdict

    # 1. group atoms by element on both sides
    r_by_el = defaultdict(list)
    p_by_el = defaultdict(list)
    for idx, sym in atomR.items():
        r_by_el[sym].append(idx)
    for idx, sym in atomP.items():
        p_by_el[sym].append(idx)

    # 2. reject if sides aren't atom-balanced by element
    if sorted(r_by_el.keys()) != sorted(p_by_el.keys()):
        return None
    for el in r_by_el:
        if len(r_by_el[el]) != len(p_by_el.get(el, [])):
            return None

    # 3. compute how many permutations we'd enumerate; bail early if too many
    total_perms = 1
    for el in r_by_el:
        total_perms *= factorial(len(r_by_el[el]))
        if total_perms > perm_cap:
            return None

    # 4. enumerate.  For each element group independently we permute the
    #    reactant positions paired with the fixed product positions.
    elements = sorted(r_by_el.keys())

    def gen_mappings(prefix, el_idx):
        if el_idx == len(elements):
            yield prefix
            return
        el = elements[el_idx]
        r_group = r_by_el[el]
        p_group = p_by_el[el]
        for perm in permutations(r_group):
            yield from gen_mappings(prefix + list(zip(perm, p_group)),
                                    el_idx + 1)

    best = None
    for tup in gen_mappings([], 0):
        try:
            broken_list, formed_list, order_changes, total_units = \
                edits_for_mapping(tup)
        except Exception:
            continue
        n_broken = len(broken_list)
        n_formed = len(formed_list)
        n_order  = len(order_changes)
        key = (total_units,
               n_broken + n_formed,
               _elementary_tiebreak(n_broken, n_formed,
                                    broken_list, formed_list),
               n_broken, n_formed, n_order)
        if best is None or key < best[0]:
            best = (key, tup, broken_list, formed_list, order_changes)

    return best


def analyze_reaction(reaction_str: str, timeout_s: int = 5):
    """
    Same methodology as before, but returns detailed mapping/bond-change info.

    Returns a dict with:
      - allowed_reaction
      - map_status
      - atom_mapping
      - n_bonds_broken
      - n_bonds_formed
      - bonds_broken
      - bonds_formed
    """
    result = {
        "allowed_reaction": False,
        "map_status": "unprocessed",
        "atom_mapping": "",
        "n_bonds_broken": "",
        "n_bonds_formed": "",
        "bonds_broken": "",
        "bonds_formed": "",
        "n_bond_order_changes": "",
        "bond_order_changes": "",
    }

    try:
        r_side, p_side = reaction_str.split(">>", 1)
    except ValueError:
        result["map_status"] = "bad_reaction_string"
        return result

    if not r_side or not p_side:
        result["map_status"] = "empty_side"
        return result

    # Build bond sets for each side as a SINGLE molecule (matching MapIsing indexing)
    r_out = bonds_and_symbols(r_side)
    if r_out[0] is None:
        result["map_status"] = "bad_reactant_smiles"
        return result
    reactant_bonds, atomR = r_out

    p_out = bonds_and_symbols(p_side)
    if p_out[0] is None:
        result["map_status"] = "bad_product_smiles"
        return result
    product_bonds, atomP = p_out

    # Ignore coordinate metal-oxygen bonds. Single source of truth:
    # MapIsing.COORD_PAIRS_SYM; the same set is stripped from the mapping
    # objective in MolGraph, keeping objective and edit counting coherent.
    def is_coord(i, j):
        return frozenset((atomR.get(i), atomR.get(j))) in mi.COORD_PAIRS_SYM

    reactant_bonds = Counter({b: c for b, c in reactant_bonds.items()
                              if not is_coord(*b)})

    # ---- mapping with timeout
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_s)

    try:
        # ------------------------------------------------------------------
        # Collect ALL candidate mappings from ALL cliques, compute the bond
        # edit count for each, and pick the one with the minimum total edits.
        # A later clique can hold a better mapping than the first (cliques
        # with the same heavy-atom size can differ in H distribution / bond
        # order alignment), so every clique is scored.
        #
        # We classify each atom-pair difference into three categories:
        #   - broken        : pair bonded in reactant, absent in product
        #   - formed        : pair absent in reactant, bonded in product
        #   - order_change  : pair bonded on BOTH sides but with different
        #                     bond order (e.g. single -> double during a
        #                     pi-bond rearrangement, such as [C]+ -> C=O).
        #
        # Bond-order rearrangements on a preserved pair are NOT covalent
        # bond break/form events -- they are pi / lone-pair redistribution.
        # Counting them as "formed"/"broken" wrongly rejects elementary
        # steps (e.g. carbocation -> carbonate ring opening).
        # ------------------------------------------------------------------
        def edits_for_mapping(tup):
            mapping_dict = dict(tup)
            inv = {p: r for r, p in mapping_dict.items()}

            mapped = Counter()
            for (i, j), count in product_bonds.items():
                ri, rj = inv.get(i), inv.get(j)
                if ri and rj:
                    pair = (min(ri, rj), max(ri, rj))
                    if not is_coord(*pair):
                        mapped[pair] += count

            broken_list = []
            formed_list = []
            order_changes = []          # list of (pair, r_order, p_order)

            for pair in set(reactant_bonds) | set(mapped):
                r_count = reactant_bonds.get(pair, 0)
                m_count = mapped.get(pair, 0)
                if r_count > 0 and m_count == 0:
                    # Pair entirely lost.  Count by bond order so that a
                    # double (or triple) bond cleaved in one step shows up
                    # as 2 (or 3) broken bond-units -- this correctly
                    # disallows a one-step C=C cleavage as an elementary
                    # step.
                    broken_list.extend([pair] * r_count)
                elif r_count == 0 and m_count > 0:
                    # Pair entirely new.  Same bond-order weighting so that
                    # forming a double bond from nothing is 2 formed units.
                    formed_list.extend([pair] * m_count)
                elif r_count != m_count:
                    # Pair bonded on BOTH sides with different order:
                    # pi-bond / lone-pair redistribution on a preserved
                    # atom pair, NOT a covalent break/form event.
                    order_changes.append((pair, r_count, m_count))

            # Total bond-order-unit difference, kept as the primary
            # mapping-selection score so that we still prefer mappings
            # that align bond orders well (same metric as before).
            total_units = sum(
                abs(mapped.get(p, 0) - reactant_bonds.get(p, 0))
                for p in (set(reactant_bonds) | set(mapped))
            )

            return (sorted(broken_list), sorted(formed_list),
                    sorted(order_changes), total_units)

        def is_coord_p(i, j):
            return frozenset((atomP.get(i), atomP.get(j))) in mi.COORD_PAIRS_SYM

        product_bonds_cov = Counter({b: c for b, c in product_bonds.items()
                                     if not is_coord_p(*b)})

        trivial = (
            (not _has_heavy_heavy_bond(reactant_bonds, atomR))
            or (not _has_heavy_heavy_bond(product_bonds_cov, atomP))
        )

        if trivial:
            best = _trivial_map_by_permutation(
                reactant_bonds, atomR,
                product_bonds, atomP,
                edits_for_mapping,
            )
            if best is None:
                result["map_status"] = "no_mapping"
                return result
            # fall through to the common best -> info conversion below
            mp = None  # sentinel so we don't accidentally use it
            nodes = edges = cliques = None

        else:
            # ------------------------------------------------------------------
            # Non-trivial reactions go through the baseline max-clique pipeline.
            # We deliberately do NOT wrap the SA sampler in a try/except with a
            # singleton-clique fallback: on large modular products, falling
            # through to enumerating every node as its own clique can burn
            # many minutes per reaction inside permutation_complete, which
            # grows memory and triggers SLURM OOM-kills that cascade to
            # pool_broken rows on the whole pool.  If the SA sampler actually
            # errors, we'd rather surface a single clean map_status='error'
            # row (via the outer try/except) than drag the pool down.
            # ------------------------------------------------------------------
            mp = mi.Mapping(reaction_str)
            nodes, edges = mp.modular_product()
 
            graph = nx.Graph()
            graph.add_nodes_from(range(len(nodes)))
            graph.add_edges_from(edges)
            mcf = ow.MaxCliques(graph)

            # Exact deterministic enumeration first (a 15 s soft budget): the
            # unseeded SA occasionally misses the true maximum clique on
            # borderline graphs (~1-1.5% unstable verdicts on controls,
            # persisting at tightened epsilon); the exact branch-and-bound
            # eliminates that class and is typically ~10x FASTER on ordinary
            # modular products. Dense pathological graphs (measured: 25k-51k
            # edges explode exponentially regardless of node count) abort at
            # the budget and fall back to SA (the pre-fix behavior).
            if graph.number_of_nodes() > 0:
                cliques, _ = mcf.find_maximum_cliques_cp(time_budget_s=15)
                if cliques is None:
                    cliques, _ = mcf.find_maximum_cliques_sa()
            else:
                cliques, _ = mcf.find_maximum_cliques_sa()

            best = None

            for cq in cliques:
                maps1 = mp.cliques_to_mappings(nodes, [cq])
                if not maps1:
                    continue
                maps2 = mp.non_equivalent(maps1)
                if not maps2:
                    continue
                maps3 = mp.filtering(maps2, "filter1")
                if not maps3:
                    continue

                for tup in maps3:
                    try:
                        broken_list, formed_list, order_changes, total_units = \
                            edits_for_mapping(tup)
                    except Exception:
                        continue
                    n_broken = len(broken_list)
                    n_formed = len(formed_list)
                    n_order  = len(order_changes)
                    # Selection key:
                    #   1) minimize total bond-order-unit difference,
                    #   2) then minimize number of true break+form events,
                    #   3) then prefer an ELEMENTARY mapping among cost ties
                    #      (see _elementary_tiebreak; makes the verdict
                    #      independent of clique enumeration order),
                    #   4) then individual break / form / order-change counts.
                    key = (total_units,
                           n_broken + n_formed,
                           _elementary_tiebreak(n_broken, n_formed,
                                                broken_list, formed_list),
                           n_broken, n_formed, n_order)
                    if best is None or key < best[0]:
                        best = (key, tup,
                                broken_list, formed_list, order_changes)

            if best is None:
                # Fallback: degenerate modular products (e.g. a single
                # homonuclear conserved bond, whose element-symmetric clique
                # yields an empty premap that permutation_complete skips) can
                # produce zero completed mappings even though a minimal-edit
                # mapping exists. Enumerate element-preserving permutations
                # directly, SIZE-GATED so large reactions cannot trigger the
                # OOM cascade the comment above warns about.
                from math import factorial
                n_perms = 1
                for c in Counter(atomR.values()).values():
                    n_perms *= factorial(c)
                if n_perms <= 40320:
                    best = _trivial_map_by_permutation(
                        reactant_bonds, atomR,
                        product_bonds, atomP,
                        edits_for_mapping,
                    )

        if best is None:
            result["map_status"] = "no_mapping"
            return result

        _, tup, broken_list, formed_list, order_changes = best
        n_broken = len(broken_list)
        n_formed = len(formed_list)
        n_order  = len(order_changes)

        info = {
            "allowed_reaction": False,
            "map_status": "mapped",
            "atom_mapping": serialize_mapping(tup, atomR, atomP),
            "n_bonds_broken": n_broken,
            "n_bonds_formed": n_formed,
            "bonds_broken": serialize_bond_list(broken_list, atomR),
            "bonds_formed": serialize_bond_list(formed_list, atomR),
            "n_bond_order_changes": n_order,
            "bond_order_changes": serialize_order_changes(order_changes, atomR),
        }

        # Elementary-step criterion:
        #   Count covalent break/form events only.  Bond-order changes on
        #   preserved pairs (pi-bond redistribution) are rearrangements
        #   within the same reaction center and are NOT counted against
        #   the step.
        #
        #   Allowed patterns (break, form):
        #     (0, 0), (0, 1), (1, 0),
        #     (1, 1) with the broken and formed pairs sharing at least
        #     one atom (the reaction center).
        allowed = False
        if n_broken == 0 and n_formed == 0:
            allowed = True
        elif n_broken == 0 and n_formed == 1:
            allowed = True
        elif n_broken == 1 and n_formed == 0:
            allowed = True
        elif n_broken == 1 and n_formed == 1:
            if set(broken_list[0]) & set(formed_list[0]):
                allowed = True

        info["allowed_reaction"] = allowed
        return info

    except _MapTimeout:
        result["allowed_reaction"] = True   # preserving your current behavior
        result["map_status"] = "timeout"
        return result

    except Exception:
        result["map_status"] = "error"
        return result

    finally:
        signal.alarm(0)


def _error_info(status):
    return {
        "allowed_reaction": False,
        "map_status": status,
        "atom_mapping": "",
        "n_bonds_broken": "",
        "n_bonds_formed": "",
        "bonds_broken": "",
        "bonds_formed": "",
        "n_bond_order_changes": "",
        "bond_order_changes": "",
    }


def _install_mem_cap(mem_cap_gb):
    """Cap this process's virtual-memory use. Excess allocations raise
    MemoryError instead of getting OOM-killed by SLURM.  No-op if 0/None."""
    if not mem_cap_gb or mem_cap_gb <= 0:
        return
    limit = int(mem_cap_gb * 1024 ** 3)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, OSError):
        # Some systems reject RLIMIT_AS; silently continue.
        pass


def worker(row, timeout_s=5, mem_cap_gb=0):
    # Installed once per worker process (the first call in that process).
    # Cheap to call repeatedly -- setrlimit on already-capped value is a no-op.
    _install_mem_cap(mem_cap_gb)

    rid, rxn = row
    try:
        info = analyze_reaction(rxn, timeout_s=timeout_s)
    except MemoryError:
        info = _error_info("memory_exceeded")
    except Exception:
        info = _error_info("worker_error")

    info["reaction_id"] = rid
    info["reaction_smiles"] = rxn
    return info


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Map all reactions and save mapping/bond-change info for every reaction."
    )
    ap.add_argument("--in_csv", required=True, help="CSV with columns: reaction_id,reaction_smiles")
    ap.add_argument("--out_csv", default="all_reactions_with_mapping_info.csv",
                    help="Output CSV with mapping info for all reactions")
    ap.add_argument("--workers", type=int, default=int(os.getenv("MAP_WORKERS", "104")),
                    help="parallel worker processes (default 104, matches a full "
                         "full standard node).  Lower this (e.g. 52) for "
                         "memory-heavy datasets.")
    ap.add_argument("--timeout", type=int, default=int(os.getenv("MAP_TIMEOUT", "2500")))
    ap.add_argument("--chunksize", type=int, default=10000,
                    help="pandas read_csv chunk size (rows read at a time).")
    ap.add_argument("--batch_size", type=int,
                    default=int(os.getenv("MAP_BATCH_SIZE", "0")),
                    help="ProcessPoolExecutor batch size.  0 (default) = one "
                         "pool per pandas-chunk, matching the original fast "
                         "behavior.  Set to a small value (e.g. 500) to opt "
                         "into fine-grained fault isolation for memory-heavy "
                         "datasets -- this is SLOWER (each batch re-forks the "
                         "pool and re-imports every library) but contains "
                         "damage when a worker dies.")
    ap.add_argument("--mem_cap_gb", type=float,
                    default=float(os.getenv("MAP_MEM_CAP_GB", "0")),
                    help="per-worker virtual-memory cap in GB.  0 (default) "
                         "disables the cap.  RLIMIT_AS counts virtual memory "
                         "including shared libs, so a low cap (<4 GB) can "
                         "trigger spurious MemoryError from harmless code; "
                         "only enable this if you know you need it.")
    args = ap.parse_args()

    # prevent hidden thread oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    t0 = time.time()
    total = 0

    wrote_header = False

    def run_batch(sub_pairs):
        """
        Run one batch of (rid, rxn) rows through a fresh ProcessPoolExecutor.
        Returns a list of result dicts for every row.  If the pool dies mid-batch
        (BrokenProcessPool -- usually SLURM OOM-kill), we record 'pool_broken'
        rows for whichever futures hadn't come back and continue to the next
        batch rather than aborting the whole chunk.
        """
        rows = []
        fut_to_row = {}
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for row in sub_pairs:
                    fut = ex.submit(worker, row, args.timeout, args.mem_cap_gb)
                    fut_to_row[fut] = row
                for f in as_completed(fut_to_row):
                    try:
                        rows.append(f.result())
                    except MemoryError:
                        rid, rxn = fut_to_row[f]
                        info = _error_info("memory_exceeded")
                        info["reaction_id"] = rid
                        info["reaction_smiles"] = rxn
                        rows.append(info)
                    except BrokenProcessPool:
                        # The pool itself died. Break out and handle below.
                        raise
                    except Exception:
                        rid, rxn = fut_to_row[f]
                        info = _error_info("future_error")
                        info["reaction_id"] = rid
                        info["reaction_smiles"] = rxn
                        rows.append(info)
        except BrokenProcessPool:
            # Worker was OOM-killed (or similar) while others were mid-flight.
            # Mark every row we didn't already record as 'pool_broken'.
            done_ids = {r["reaction_id"] for r in rows}
            for row in sub_pairs:
                rid, rxn = row
                if rid in done_ids:
                    continue
                info = _error_info("pool_broken")
                info["reaction_id"] = rid
                info["reaction_smiles"] = rxn
                rows.append(info)
        return rows

    with open(args.out_csv, "w") as fout:
        for chunk in pd.read_csv(
            args.in_csv,
            usecols=["reaction_id", "reaction_smiles"],
            chunksize=args.chunksize
        ):
            pairs = list(zip(
                chunk["reaction_id"].astype(int).tolist(),
                chunk["reaction_smiles"].astype(str).tolist()
            ))
            total += len(pairs)

            rows_out = []
            # batch_size <= 0 => one pool for the whole pandas-chunk (fast,
            # original behavior).  Positive => fine-grained batches
            # (slower, better fault isolation).
            if args.batch_size and args.batch_size > 0:
                bs = args.batch_size
                for bstart in range(0, len(pairs), bs):
                    sub = pairs[bstart:bstart + bs]
                    rows_out.extend(run_batch(sub))
            else:
                rows_out.extend(run_batch(pairs))

            out_df = pd.DataFrame(rows_out, columns=[
                "reaction_id",
                "reaction_smiles",
                "allowed_reaction",
                "map_status",
                "atom_mapping",
                "n_bonds_broken",
                "n_bonds_formed",
                "bonds_broken",
                "bonds_formed",
                "n_bond_order_changes",
                "bond_order_changes",
            ])

            out_df.sort_values("reaction_id", inplace=True)

            out_df.to_csv(
                fout,
                index=False,
                header=not wrote_header
            )
            wrote_header = True

            n_allowed = int(out_df["allowed_reaction"].fillna(False).sum())
            print(
                f"[progress] processed={total}  allowed={n_allowed} (this chunk)  "
                f"elapsed={time.time()-t0:.1f}s",
                end="\r"
            )

    print(f"\nDone. Processed {total} reactions. Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
