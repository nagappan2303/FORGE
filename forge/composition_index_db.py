"""Disk-resident composition index: memory-bounded enumeration.

FORGE's seeded enumerator finds product pairs by a `composition -> pairs` index
(`build_composition_indexes` in seeded_reaction_generator). That index is
O(N^2) and, under fork OR mpi, gets replicated PER WORKER (~3.5 GB x ranks for
chemistries), which forces multi-node just to hold copies of the same table.

This module moves the index ONTO DISK (one sqlite, built once) so every worker
holds only mol_entries + a read-only connection (~30 MB), independent of species
count. Single-node-many-workers then fits a large
chemistry on one node. It is also
built ONCE and reused every iteration (the index is a pure function of
mol_entries).

The enumeration result is IDENTICAL to the in-memory path: same index contents,
same tree walk, same rows, verified row-for-row against the in-memory path
during development.

Schema:
  compositions(comp_id INTEGER PRIMARY KEY, comp_key TEXT UNIQUE)
  unary_index(comp_id INTEGER, species_ind INTEGER)            -- INDEX(comp_id)
  pair_index(comp_id INTEGER, ind_i INTEGER, ind_j INTEGER)    -- INDEX(comp_id)
"""
from __future__ import annotations

import os
import sqlite3
from typing import List, Tuple

from forge.seeded_reaction_generator import walk_tree, _build_row_partial
from chemistry_lib.constants import Terminal


def _comp_key(species_seq) -> str:
    return "_".join(sorted(species_seq))


def build_index_db(mol_entries, db_path: str, commit_every: int = 200000) -> dict:
    """Build the on-disk composition index. One-time; pure function of
    mol_entries. Returns {n_compositions, n_unary, n_pairs}."""
    if os.path.exists(db_path):
        os.remove(db_path)
    species_tup = [tuple(sorted(e.species)) for e in mol_entries]

    con = sqlite3.connect(db_path)

    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("CREATE TABLE compositions(comp_id INTEGER PRIMARY KEY, comp_key TEXT UNIQUE)")
    con.execute("CREATE TABLE unary_index(comp_id INTEGER, species_ind INTEGER)")
    con.execute("CREATE TABLE pair_index(comp_id INTEGER, ind_i INTEGER, ind_j INTEGER)")

    comp_id: dict = {}

    def cid(key: str) -> int:
        c = comp_id.get(key)
        if c is None:
            c = len(comp_id)
            comp_id[key] = c
            con.execute("INSERT INTO compositions VALUES (?,?)", (c, key))
        return c

    # unary
    ubuf = []
    for i, e in enumerate(mol_entries):
        ubuf.append((cid("_".join(species_tup[i])), e.ind))
    con.executemany("INSERT INTO unary_index VALUES (?,?)", ubuf)
    n_unary = len(ubuf)

    # pairs (i <= j), mirroring build_composition_indexes exactly
    n = len(mol_entries)
    pbuf = []
    n_pairs = 0
    for i in range(n):
        si = species_tup[i]
        ind_i = mol_entries[i].ind
        for j in range(i, n):
            key = "_".join(sorted(si + species_tup[j]))
            pbuf.append((cid(key), ind_i, mol_entries[j].ind))
            if len(pbuf) >= commit_every:
                con.executemany("INSERT INTO pair_index VALUES (?,?,?)", pbuf)
                n_pairs += len(pbuf); pbuf.clear()
    if pbuf:
        con.executemany("INSERT INTO pair_index VALUES (?,?,?)", pbuf)
        n_pairs += len(pbuf)

    con.execute("CREATE INDEX ix_unary ON unary_index(comp_id)")
    con.execute("CREATE INDEX ix_pair ON pair_index(comp_id)")
    con.commit(); con.close()
    return {"n_compositions": len(comp_id), "n_unary": n_unary, "n_pairs": n_pairs}


class DiskCompositionIndex:
    """Read-only per-worker handle. Holds only a sqlite connection + a tiny
    comp_key->comp_id cache; never the full pair table in RAM."""

    def __init__(self, db_path: str):
        self.con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.cur = self.con.cursor()
        self._cid: dict = {}

    def _comp_id(self, comp_key: str):
        if comp_key in self._cid:
            return self._cid[comp_key]
        r = self.cur.execute(
            "SELECT comp_id FROM compositions WHERE comp_key=?", (comp_key,)).fetchone()
        cid = r[0] if r else None
        self._cid[comp_key] = cid
        return cid

    def unary(self, comp_key: str) -> List[int]:
        cid = self._comp_id(comp_key)
        if cid is None:
            return []
        return [row[0] for row in self.cur.execute(
            "SELECT species_ind FROM unary_index WHERE comp_id=?", (cid,))]

    def pairs(self, comp_key: str) -> List[Tuple[int, int]]:
        cid = self._comp_id(comp_key)
        if cid is None:
            return []
        return self.cur.execute(
            "SELECT ind_i, ind_j FROM pair_index WHERE comp_id=?", (cid,)).fetchall()

    def close(self):
        try:
            self.con.close()
        except Exception:
            pass


def process_bucket_entry_disk(task, mol_entries, species_tup, charges, G_solv,
                              tree, params, disk_index: DiskCompositionIndex):
    """Disk-index analogue of seeded_reaction_generator._process_bucket_entry,
    STRUCTURAL-ONLY (no inline AAM; the decoupled flow does AAM as a batch).
    Returns (kept_rows, n_tested). Rows have the same 10-tuple shape.
    """
    r1, r2, skip_self_reactions = task
    T = params.get("temperature", 298.15)
    efe = params.get("electron_free_energy", 0.0)

    n_r = 1 if r2 == -1 else 2
    if n_r == 1:
        comp_seq = species_tup[r1]
        reactants = [r1]
    else:
        comp_seq = tuple(sorted(list(species_tup[r1]) + list(species_tup[r2])))
        reactants = [r1, r2]
    comp_key = "_".join(comp_seq)
    r_sorted = tuple(sorted(reactants))

    n_tested = 0
    kept_rows = []

    for p1 in disk_index.unary(comp_key):
        products = [p1]
        if skip_self_reactions and tuple(sorted(products)) == r_sorted:
            continue
        reaction = {"reactants": reactants, "products": products,
                    "number_of_reactants": n_r, "number_of_products": 1}
        n_tested += 1
        if walk_tree(tree, reaction, mol_entries, params) != Terminal.KEEP:
            continue
        kept_rows.append(_build_row_partial(reactants, products, reaction,
                                            charges, G_solv, efe, T))

    for p1, p2 in disk_index.pairs(comp_key):
        products = [p1, p2]
        if skip_self_reactions and tuple(sorted(products)) == r_sorted:
            continue
        reaction = {"reactants": reactants, "products": products,
                    "number_of_reactants": n_r, "number_of_products": 2}
        n_tested += 1
        if walk_tree(tree, reaction, mol_entries, params) != Terminal.KEEP:
            continue
        kept_rows.append(_build_row_partial(reactants, products, reaction,
                                            charges, G_solv, efe, T))

    return kept_rows, n_tested
