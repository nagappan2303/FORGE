"""Ising-formulation atom-atom mapper.

The Ising formulation and annealing core follow the atom-mapping method of
Ali et al. This copy departs from the reference implementation in four ways:
explicit hydrogen atoms are graph nodes (RDKit AddHs replaces CGRtools),
reactions with an empty heavy-atom bond graph on either side take a direct
enumeration fast path (reaction_mapper._trivial_map_by_permutation), each
mapper call runs in a SIGKILL-isolated subprocess with a hard timeout
(forge.aam_filter), and verdicts are memoized in a canonical reaction-SMILES
cache (forge.aam_cache).
"""
import time
from itertools import combinations, permutations
from copy import deepcopy
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem

COORD_PAIRS_AN  = frozenset({frozenset((8, 3)),        # O-Li
                             frozenset((8, 11))})      # O-Na
COORD_PAIRS_SYM = frozenset({frozenset(("O", "Li")),
                             frozenset(("O", "Na"))})


class MolGraph:
    """
    Molecular graph including explicit hydrogen atoms.
    Uses RDKit instead of CGRtools so that H atoms are first-class graph nodes.

    Atom indices are 1-based (matching the original convention).
    """

    def __init__(self, strn):

        mol = Chem.MolFromSmiles(strn)
        if mol is None:
            raise ValueError(f"RDKit cannot parse SMILES: {strn}")

        Chem.Kekulize(mol, clearAromaticFlags=True)
        mol = Chem.AddHs(mol)                       # <<< explicit H

        self._mol = mol

        # --- atom info (1-based index) ---
        atom_info = []
        for a in mol.GetAtoms():
            idx = a.GetIdx() + 1                     # 1-based
            an  = a.GetAtomicNum()
            atom_info.append((idx, an))

        self.atoms = sorted([idx for idx, _ in atom_info])
        self.an    = {idx: an for idx, an in atom_info}

        bond_info = []
        for i, b in enumerate(mol.GetBonds()):
            x = b.GetBeginAtomIdx() + 1
            y = b.GetEndAtomIdx()   + 1
            if x > y:
                x, y = y, x
            if frozenset((self.an[x], self.an[y])) in COORD_PAIRS_AN:
                continue
            order = int(b.GetBondTypeAsDouble())
            bond_info.append((i, (x, y), order))

        self.bonds   = [pair  for _, pair, _     in bond_info]
        self.bn_an   = [(self.an[x], self.an[y]) for x, y in self.bonds]
        self.order   = [z     for _, _, z        in bond_info]
        self.bn_kind = [
            (x, y) if x <= y else (y, x)
            for x, y in self.bn_an
        ]

        # dic_hyd: number of H neighbours for every atom
        # (With explicit H this is still useful for backward compatibility of score/isomorphism.)
        self.dic_hyd = {}
        for idx in self.atoms:
            if self.an[idx] == 1:
                self.dic_hyd[idx] = 0          # H itself has 0 H-neighbours
            else:
                self.dic_hyd[idx] = sum(
                    1 for (x, y) in self.bonds
                    if (x == idx and self.an[y] == 1) or
                       (y == idx and self.an[x] == 1)
                )

        # --- neighbourhood dict ---
        self.dic_neib = defaultdict(list)
        for x, y in self.bonds:
            self.dic_neib[x].append(y)
            self.dic_neib[y].append(x)
        self.dic_neib = dict(self.dic_neib)
        for n in self.atoms:
            self.dic_neib.setdefault(n, [])

        # --- bond-order lookup (both directions) ---
        self.dic_order = {}
        # build fast set for O(1) lookup
        _bond_set = {}
        for bi, (x, y) in enumerate(self.bonds):
            _bond_set[(x, y)] = bi
        for x, y in combinations(self.atoms, 2):
            if x > y:
                x, y = y, x
            if (x, y) in _bond_set:
                o = self.order[_bond_set[(x, y)]]
            else:
                o = 0
            self.dic_order[(x, y)] = o
            self.dic_order[(y, x)] = o

        # heavy-atom parent for every hydrogen (used by smart_permutation_complete)
        self.h_parent = {}
        for idx in self.atoms:
            if self.an[idx] == 1:
                # H is bonded to exactly one atom
                neibs = self.dic_neib[idx]
                if neibs:
                    self.h_parent[idx] = neibs[0]

    # -----------------------------------------------------------------
    def refer_bond(self, n, m):
        if n > m:
            n, m = m, n
        if (n, m) in self.bonds:
            b_num = self.bonds.index((n, m))
            return self.bn_kind[b_num]
        return None

    def refer_bond_order(self, n, m):
        if n > m:
            n, m = m, n
        if (n, m) in self.bonds:
            return self.order[self.bonds.index((n, m))]
        return 0

    def bond_common_atom(self, n, m):
        n1, n2 = self.bonds[n]
        m1, m2 = self.bonds[m]
        if n1 == m1 or n1 == m2:
            return self.an[n1]
        if n2 == m1 or n2 == m2:
            return self.an[n2]
        return None

    def bond_mapping(self, n, m):
        n1, n2 = self.bonds[n]
        m1, m2 = self.bonds[m]
        if n1 == m1:
            return n2, n1, m2
        if n1 == m2:
            return n2, n1, m1
        if n2 == m1:
            return n1, n2, m2
        if n2 == m2:
            return n1, n2, m1
        return None


# =====================================================================
class Mapping:

    def __init__(self, str_smiles):

        sm_rct, sm_prd = str_smiles.split('>>')

        self.rct = MolGraph(sm_rct)
        self.prd = MolGraph(sm_prd)

    # -----------------------------------------------------------------
    def unmapped_area(self, mp):
        if mp == []:
            return [], []
        r_mapped, p_mapped = zip(*mp)
        r_unmap = [x for x in self.rct.atoms if x not in r_mapped]
        p_unmap = [x for x in self.prd.atoms if x not in p_mapped]
        return r_unmap, p_unmap

    # -----------------------------------------------------------------
    def modular_product(self, mode='bond', premap=[]):
        """
        Build the modular product graph using HEAVY ATOMS ONLY.
        H atoms inflate the graph enormously (every C-H bond pairs with every
        other C-H bond, etc.) and make max-clique infeasible.
        H atoms are still present in self.atoms and get assigned later by
        permutation_complete's smart parent-based greedy.
        """
        r_area, p_area = self.unmapped_area(premap)

        r_an = self.rct.an
        p_an = self.prd.an

        if mode == 'atom':
            nodes = []
            for i_idx in self.rct.atoms:
                if r_an[i_idx] == 1:           # skip H
                    continue
                for j_idx in self.prd.atoms:
                    if p_an[j_idx] == 1:       # skip H
                        continue
                    if r_an[i_idx] == p_an[j_idx]:
                        if r_area == [] and p_area == []:
                            nodes.append((i_idx, j_idx))
                        elif (i_idx in r_area) and (j_idx in p_area):
                            nodes.append((i_idx, j_idx))

            edges = []
            node_idx = {n: i for i, n in enumerate(nodes)}
            for node1, node2 in combinations(nodes, 2):
                r1, p1 = node1
                r2, p2 = node2
                if r1 != r2 and p1 != p2:
                    r_bond = self.rct.refer_bond(r1, r2)
                    p_bond = self.prd.refer_bond(p1, p2)
                    if r_bond == p_bond:
                        edges.append((node_idx[node1], node_idx[node2]))

        if mode == 'bond':
            # Heavy-atom-only bond lists
            r_heavy_bidx = [i for i, (x, y) in enumerate(self.rct.bonds)
                            if r_an[x] != 1 and r_an[y] != 1]
            p_heavy_bidx = [i for i, (x, y) in enumerate(self.prd.bonds)
                            if p_an[x] != 1 and p_an[y] != 1]

            rb_neib = {i: (r_an[self.rct.bonds[i][0]], r_an[self.rct.bonds[i][1]])
                       for i in r_heavy_bidx}
            pb_neib = {i: (p_an[self.prd.bonds[i][0]], p_an[self.prd.bonds[i][1]])
                       for i in p_heavy_bidx}

            nodes = []
            for i in r_heavy_bidx:
                x, y = rb_neib[i]
                for j in p_heavy_bidx:
                    v, w = pb_neib[j]
                    if (x == v and y == w) or (x == w and y == v):
                        if r_area == [] and p_area == []:
                            nodes.append((i, j))
                        else:
                            r1, r2 = self.rct.bonds[i]
                            p1, p2 = self.prd.bonds[j]
                            if (r1 in r_area) and (r2 in r_area) and (p1 in p_area) and (p2 in p_area):
                                nodes.append((i, j))

            edges = []
            node_idx = {n: i for i, n in enumerate(nodes)}
            for node1, node2 in combinations(nodes, 2):
                r1, p1 = node1
                r2, p2 = node2
                if r1 != r2 and p1 != p2:
                    if self.rct.bond_common_atom(r1, r2) == self.prd.bond_common_atom(p1, p2):
                        edges.append((node_idx[node1], node_idx[node2]))

        return nodes, edges

    # -----------------------------------------------------------------
    def to_atom_mapping(self, bond_map):
        atom_map = []

        for n, m in bond_map:
            r1, r2 = self.rct.bonds[n]
            p1, p2 = self.prd.bonds[m]
            r_an1, r_an2 = self.rct.bn_an[n]
            p_an1, p_an2 = self.prd.bn_an[m]

            if r_an1 != r_an2 and p_an1 != p_an2:
                if r_an1 == p_an1 and r_an2 == p_an2:
                    atom_map += [(r1, p1), (r2, p2)]
                if r_an1 == p_an2 and r_an2 == p_an1:
                    atom_map += [(r1, p2), (r2, p1)]

        for (r1, p1), (r2, p2) in combinations(bond_map, 2):
            if self.rct.bond_common_atom(r1, r2):
                rx, ry, rz = self.rct.bond_mapping(r1, r2)
                px, py, pz = self.prd.bond_mapping(p1, p2)
                atom_map += [(rx, px), (ry, py), (rz, pz)]

        atom_map = list(set(atom_map))
        atom_map.sort()
        return atom_map

    # -----------------------------------------------------------------
    def permutation_complete(self, premaps=[]):
        """
        Smart completion that avoids H-permutation explosion.

        Strategy:
          1. Group unmapped atoms by element.
          2. For heavy atoms (non-H), try all permutations within each element
             group (these groups are typically tiny after max-clique mapping).
          3. For H atoms, use a greedy parent-based assignment:
             Each unmapped H on reactant side is matched to an unmapped H on the
             product side whose parent heavy atom is the image of the reactant H's
             parent under the (already determined) heavy-atom mapping.
             This is O(n) instead of O(n!).
          4. If greedy H assignment fails (parent not mapped, or no product H
             available), fall back to element-preserving permutation for the
             remaining few H atoms only.
        """
        total_added = []

        for x in premaps:
            if not x:
                continue

            r_mapped, p_mapped = zip(*x)
            r_mapped_set = set(r_mapped)
            p_mapped_set = set(p_mapped)

            r_rsd = [a for a in self.rct.atoms if a not in r_mapped_set]
            p_rsd = [a for a in self.prd.atoms if a not in p_mapped_set]

            if not r_rsd and not p_rsd:
                total_added.append(x)
                continue

            # Split residuals into H and non-H
            r_heavy = [a for a in r_rsd if self.rct.an[a] != 1]
            p_heavy = [a for a in p_rsd if self.prd.an[a] != 1]
            r_h     = [a for a in r_rsd if self.rct.an[a] == 1]
            p_h     = [a for a in p_rsd if self.prd.an[a] == 1]

            # Group heavy residuals by element
            r_heavy_by_el = defaultdict(list)
            p_heavy_by_el = defaultdict(list)
            for a in r_heavy:
                r_heavy_by_el[self.rct.an[a]].append(a)
            for a in p_heavy:
                p_heavy_by_el[self.prd.an[a]].append(a)

            # Check element counts match
            if sorted(r_heavy_by_el.keys()) != sorted(p_heavy_by_el.keys()):
                continue
            if any(len(r_heavy_by_el[el]) != len(p_heavy_by_el.get(el, []))
                   for el in r_heavy_by_el):
                continue
            if len(r_h) != len(p_h):
                continue

            # Generate all heavy-atom permutations (should be small, typically 0-3)
            heavy_perms = [{}]
            for el in r_heavy_by_el:
                r_group = r_heavy_by_el[el]
                p_group = p_heavy_by_el[el]
                new_perms = []
                for perm in permutations(r_group):
                    partial = dict(zip(perm, p_group))
                    for existing in heavy_perms:
                        merged = {**existing, **partial}
                        new_perms.append(merged)
                heavy_perms = new_perms

            # For each heavy permutation, do greedy H assignment
            for hperm in heavy_perms:
                # Full heavy-atom mapping: existing + this permutation's heavy part
                heavy_map = dict(x)        # r -> p
                heavy_map.update(hperm)

                # Greedy H assignment based on parent mapping
                p_h_available = set(p_h)
                h_assignment = {}
                unmatched_r_h = []

                # Build product-side lookup: for each (heavy_parent, element=H) -> list of H indices
                p_h_by_parent = defaultdict(list)
                for ph in p_h:
                    parent = self.prd.h_parent.get(ph)
                    if parent is not None:
                        p_h_by_parent[parent].append(ph)

                for rh in r_h:
                    r_parent = self.rct.h_parent.get(rh)
                    if r_parent is not None:
                        p_parent = heavy_map.get(r_parent)
                        if p_parent is not None:
                            candidates = [ph for ph in p_h_by_parent.get(p_parent, [])
                                          if ph in p_h_available]
                            if candidates:
                                chosen = candidates[0]
                                h_assignment[rh] = chosen
                                p_h_available.discard(chosen)
                                continue
                    unmatched_r_h.append(rh)

                unmatched_p_h = sorted(p_h_available)

                if len(unmatched_r_h) != len(unmatched_p_h):
                    continue

                # For unmatched H (typically 0-4), try permutations
                if len(unmatched_r_h) == 0:
                    additional = list(hperm.items()) + list(h_assignment.items())
                    total_added.append(x + additional)
                elif len(unmatched_r_h) <= 8:
                    for perm in permutations(unmatched_r_h):
                        additional = (list(hperm.items()) +
                                      list(h_assignment.items()) +
                                      list(zip(perm, unmatched_p_h)))
                        total_added.append(x + additional)
                else:
                    # Too many unmatched H; just use first match
                    additional = (list(hperm.items()) +
                                  list(h_assignment.items()) +
                                  list(zip(unmatched_r_h, unmatched_p_h)))
                    total_added.append(x + additional)

        return total_added

    # -----------------------------------------------------------------
    def isomorphism(self, maps, mode='bond'):

        def connected_comp(idxs, adj):
            adj_dic = defaultdict(set)
            for i, j in adj:
                adj_dic[i].add(j)
                adj_dic[j].add(i)

            visited_all = set()
            components = []
            for start in idxs:
                if start in visited_all:
                    continue
                comp = set()
                queue = {start}
                while queue:
                    comp |= queue
                    nxt = set()
                    for q in queue:
                        nxt |= adj_dic.get(q, set())
                    queue = nxt - comp
                visited_all |= comp
                components.append(comp)
            return components

        if len(maps) > 10000:
            return [set(range(len(maps)))]

        r_iso = []
        p_iso = []

        if mode == 'bond':
            for mp in maps:
                r_mapped, p_mapped = zip(*mp)
                mapped_r_bonds = [(x, y, self.rct.dic_order[(x, y)])
                                  for x, y in self.rct.bonds
                                  if x in r_mapped and y in r_mapped]
                mapped_p_bonds = [(x, y, self.prd.dic_order[(x, y)])
                                  for x, y in self.prd.bonds
                                  if x in p_mapped and y in p_mapped]
                to_rct = {y: x for x, y in mp}
                to_prd = {x: y for x, y in mp}

                conv_p_bonds = []
                conv_r_bonds = []
                for x, y, order in mapped_p_bonds:
                    s, t = to_rct[x], to_rct[y]
                    if s > t:
                        s, t = t, s
                    conv_p_bonds.append((s, t, order))
                for x, y, order in mapped_r_bonds:
                    s, t = to_prd[x], to_prd[y]
                    if s > t:
                        s, t = t, s
                    conv_r_bonds.append((s, t, order))

                rs, ps = zip(*mp)
                r_Hs = [(to_prd[x], self.rct.dic_hyd[x]) for x in rs]
                p_Hs = [(to_rct[x], self.prd.dic_hyd[x]) for x in ps]
                r_Hs.sort()
                p_Hs.sort()
                conv_r_bonds.sort()
                conv_p_bonds.sort()

                r_iso.append((conv_r_bonds, r_Hs))
                p_iso.append((conv_p_bonds, p_Hs))

        connectivity = []
        for i, j in combinations(range(len(maps)), 2):
            if r_iso[i] == r_iso[j] or p_iso[i] == p_iso[j]:
                connectivity.append((i, j))

        return connected_comp(list(range(len(maps))), connectivity)

    # -----------------------------------------------------------------
    def change_bonds(self, mp):
        dic_map = dict(mp)
        cleav_bonds = []
        for x, y in self.rct.bonds:
            if x not in dic_map or y not in dic_map:
                continue
            if self.prd.refer_bond(dic_map[x], dic_map[y]) is None:
                cleav_bonds.append((x, y))

        dic_map_rev = {y: x for x, y in mp}
        form_bonds = []
        for x, y in self.prd.bonds:
            if x not in dic_map_rev or y not in dic_map_rev:
                continue
            if self.rct.refer_bond(dic_map_rev[x], dic_map_rev[y]) is None:
                form_bonds.append((x, y))

        return cleav_bonds, form_bonds

    # -----------------------------------------------------------------
    def score(self, maps, mode='bond'):
        """
        With explicit H, hydrogen-count scoring is meaningless (always 0 for
        correctly mapped molecules).  Default to bond-order scoring.
        """
        scores = []
        for mp in maps:
            sc = 0
            if mode == 'hydrogen':
                # Still works: counts H-neighbour difference (should be ~0 with explicit H)
                sc += sum(abs(self.rct.dic_hyd[x] - self.prd.dic_hyd[y]) for x, y in mp)
            elif mode == 'bond':
                for (x, y), (s, t) in combinations(mp, 2):
                    sc += abs(self.rct.dic_order.get((x, s), 0) - self.prd.dic_order.get((y, t), 0))
            elif mode == 'hydrogen + bond':
                sc += sum(abs(self.rct.dic_hyd[x] - self.prd.dic_hyd[y]) for x, y in mp)
                for (x, y), (s, t) in combinations(mp, 2):
                    sc += abs(self.rct.dic_order.get((x, s), 0) - self.prd.dic_order.get((y, t), 0))
            scores.append(sc)

        best_score = min(scores)
        return [maps[i] for i, x in enumerate(scores) if x == best_score]

    # -----------------------------------------------------------------
    def filtering(self, maps, mode):
        """
        filter1: bond-order score only (hydrogen score is useless with explicit H).
        filter2: bond-order score (same as filter1 with explicit H).
        """
        if mode in ('filter1', 'filter2'):
            return self.score(maps, 'bond')
        return maps

    # -----------------------------------------------------------------
    def cliques_to_mappings(self, nodes, cliques, mode='bond'):
        if mode == 'atom':
            maps = [[nodes[x] for x in cq] for cq in cliques]
        elif mode == 'bond':
            btb_maps = [[nodes[x] for x in cq] for cq in cliques]
            maps = [self.to_atom_mapping(x) for x in btb_maps]
        else:
            maps = []

        comp_maps = self.permutation_complete(maps)
        for x in comp_maps:
            x.sort()
        return comp_maps

    # -----------------------------------------------------------------
    def non_equivalent(self, maps):
        groups = self.isomorphism(maps, 'bond')
        return [maps[gp.pop()] for gp in groups]
