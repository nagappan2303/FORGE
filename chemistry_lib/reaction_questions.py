"""Reaction decision tree: the questions that accept or reject candidate
reactions and assign their rates. Adapted from the framework of Barter
et al. with this project's modifications."""

import math
from chemistry_lib.mol_entry import MoleculeEntry
from functools import partial
import itertools
import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from chemistry_lib.constants import Terminal, ROOM_TEMP, KB, PLANCK, m_formulas
from monty.json import MSONable
from collections import Counter

from typing import Optional
import time

from rdkit import Chem


"""
The reaction decision tree:

A question is a function q(reaction, mol_entries, params) -> Bool

reaction is a dict:

        reaction = { 'reactants' : reactant indices,
                     'products' : product indices,
                     'number_of_reactants',
                     'number_of_products'}
params is a dict:


        params = { 'temperature',
                   'electron_free_energy' }

The lists of reactant and product indices always have length two. We
use -1 when there is a only a single reactant or product.

The questions can also set reaction['rate'] and reaction['dG']

Questions will be writable by hand, or we could have machine learning
filters.

A node is either a Terminal or a non empty list [(question, node)]

class Terminal(Enum): KEEP = 1 DISCARD = -1

For the return value of a question, True means travel to this node and
False means try next question in the list.

for non terminal nodes, it is an error if every question returns
False. i.e getting stuck at a non terminal node is an error.

Once a Terminal node is reached, it tells us whether to keep or
discard the reaction.

logging decision tree: The dispatcher takes a second decision tree as
an argument, the logging decision tree. Reactions which return
Terminal.KEEP from the logging decision tree will be logged in the
generation report, with location specified by the argument
generation_report_path

"""

hydrogen_graph = nx.MultiGraph()
hydrogen_graph.add_node(0, specie='H')
hydrogen_hash = weisfeiler_lehman_graph_hash(
    hydrogen_graph,
    node_attr='specie')

fluorine_graph = nx.MultiGraph()
fluorine_graph.add_node(0, specie='F')
fluorine_hash = weisfeiler_lehman_graph_hash(
    fluorine_graph,
    node_attr='specie')


def _site_symbol(site):
    """Element symbol of a single-occupancy site, compatible with both
    older pymatgen (site.specie) and newer releases where the attribute
    was removed in favor of the species Composition."""
    sp = getattr(site, "specie", None)
    if sp is not None:
        return sp.symbol
    return site.species.elements[0].symbol

def run_decision_tree(
        reaction,
        mol_entries,
        params,
        decision_tree,
        decision_pathway=None):
    node = decision_tree

    while type(node) == list:
        next_node = None
        for (question, new_node) in node:
            if question(reaction, mol_entries, params):

                # if decision_pathway is a list,
                # append the question which
                # answered true i.e the edge we follow
                if decision_pathway is not None:
                    decision_pathway.append(question)

                next_node = new_node
                break

        node = next_node


    if type(node) == Terminal:
        if decision_pathway is not None:
            decision_pathway.append(node)

        if node == Terminal.KEEP:
            return True
        else:
            return False
    else:
        print(node)
        raise Exception(
            """
            unexpected node type reached.
            this is usually caused because none of the questions in some node returned True.
            """)



def default_rate(dG_barrier, params):
    kT = KB * params['temperature']
    max_rate = kT / PLANCK
    rate = max_rate * math.exp(- dG_barrier / kT)
    return rate

class dG_above_threshold(MSONable):

    def __init__(self, threshold, free_energy_type, constant_barrier):

        self.threshold = threshold
        self.free_energy_type = free_energy_type
        self.constant_barrier = constant_barrier

        if free_energy_type == 'free_energy':
            self.get_free_energy = lambda mol: mol.free_energy
        elif free_energy_type == 'solvation_free_energy':
            self.get_free_energy = lambda mol: mol.solvation_free_energy
        else:
            raise Exception("unrecognized free energy type")

    def __str__(self):
        return (
            self.free_energy_type +
            " dG is above threshold=" +
            str(self.threshold))

    def __call__(self, reaction, mol_entries, params):


        dG = 0.0

        # positive dCharge means electrons are lost
        dCharge = 0.0
        #print("Reaction details:")
        #print("Reactants:")
        for i in range(reaction['number_of_reactants']):
            reactant_index = reaction['reactants'][i]
            mol = mol_entries[reactant_index]
            dG -= self.get_free_energy(mol)
            dCharge -= mol.charge
            #print(mol)
            #print(self.get_free_energy(mol))
        
        #print("Products:")
        for j in range(reaction['number_of_products']):
            product_index = reaction['products'][j]
            mol = mol_entries[product_index]
            dG += self.get_free_energy(mol)
            dCharge += mol.charge
            #print(mol)
            #print(self.get_free_energy(mol))

        #print("dG before electron free energy correction: ",dG)
        dG += dCharge * params['electron_free_energy']
        #print("dG after electron free energy correction: ",dG)
        #print("\n\n")
        if dG > self.threshold:
            return True
        else:
            reaction['dG'] = dG
            if dG < 0:
                barrier = self.constant_barrier
            else:
                barrier = dG + self.constant_barrier

            reaction['dG_barrier'] = barrier
            reaction['rate'] = default_rate(barrier, params)
            return False


class is_redox_reaction(MSONable):

    def __init__(self):
        pass

    def __str__(self):
        return "is redox reaction"

    def __call__(self, reaction, mol_entries, params):
        # positive dCharge means electrons are lost
        dCharge = 0.0

        for i in range(reaction['number_of_reactants']):
            reactant_index = reaction['reactants'][i]
            mol = mol_entries[reactant_index]
            dCharge -= mol.charge

        for j in range(reaction['number_of_products']):
            product_index = reaction['products'][j]
            mol = mol_entries[product_index]
            dCharge += mol.charge

        if dCharge == 0:
            reaction['is_redox'] = False
            return False
        else:
            reaction['is_redox'] = True
            return True


class too_many_reactants_or_products(MSONable):
    def __init__(self):
        pass

    def __str__(self):
        return "too many reactants or products"


    def __call__(self, reaction, mols, params):
        if (reaction['number_of_reactants'] != 1 or
            reaction['number_of_products'] != 1):
            return True
        else:
            return False


class metal_metal_reaction(MSONable):
    def __init__(self):
        pass

    def __call__(self, reaction, mol_entries, params):
        if (reaction['number_of_reactants'] == 1 and
            reaction['number_of_products'] == 1 and
            (mol_entries[reaction['reactants'][0]].formula in m_formulas) and
            (mol_entries[reaction['products'][0]].formula in m_formulas)):

            return True
        else:
            return False


class dcharge_too_large(MSONable):
    def __init__(self):
        pass

    def __str__(self):
        return "change in charge is too large"

    def __call__(self, reaction, mol_entries, params):
        dCharge = 0.0

        for i in range(reaction['number_of_reactants']):
            reactant_index = reaction['reactants'][i]
            mol = mol_entries[reactant_index]
            dCharge -= mol.charge

        for j in range(reaction['number_of_products']):
            product_index = reaction['products'][j]
            mol = mol_entries[product_index]
            dCharge += mol.charge

        if abs(dCharge) > 1:
            return True
        else:
            return False



def marcus_barrier(reaction, mols, params):

    """
        Okay, so Marcus Theory.The math works out like so.∆G* = λ/4 (1 +
    ∆G / λ)^2 ∆G is the Gibbs free energy of the reaction, ∆G* is the
    energy barrier, and λ is the “reorganization energy” (basically the
    energy penalty for reorganizing the solvent environment to accommodate
    the change in local charge).The reorganization energy can be broken up
    into two terms, an inner term (“i”) representing the contribution from
    the first solvation shell and an outer term (“o”) representing the
    contribution from the bulk solvent: λ = λi + λoλo = ∆e/(8 pi ε0) (1/r
    - 1/R) (1/n^2 - 1/ε) where ∆e is the change in charge in terms of
    fundamental charge (1.602 * 10 ^-19 C), ε0 is the vacuum permittivity
    (8.854 * 10 ^-12 F/m), r is the first solvation shell radius (I
    usually just pick a constant, say 6 Angstrom), R is the distance to
    the electrode (again, for these purposes, just pick something - say
    7.5 Angstrom), n is the index of refraction (1.415 for EC) and ε is
    the relative dielectric (18.5 for EC/EMC).
    """

    reactant = mols[reaction['reactants'][0]]
    product = mols[reaction['products'][0]]
    dCharge = product.charge - reactant.charge
    n = 1.415  # index of refraction; variable
    eps = 18.5  # dielectric constant; variable

    r = 6.0  # in Angstrom
    R = 7.5  # in Angstrom

    eps_0 = 8.85419 * 10 ** -12  # vacuum permittivity
    e = 1.602 * 10 ** -19  # fundamental charge

    l_outer = e / (8 * math.pi * eps_0)
    l_outer *= (1 / r - 1/(2 * R)) * 10 ** 10  # Converting to SI units; factor of 2 is because of different definitions of the distance to electrode
    l_outer *= (1 / n ** 2 - 1 / eps)

    if dCharge == -1:
        vals = [reactant.electron_affinity, product.ionization_energy]
        vals_filtered = [v for v in vals if v is not None]
        l_inner = sum(vals_filtered) / len(vals_filtered)

    if dCharge == 1:
        vals = [reactant.ionization_energy, product.electron_affinity]
        vals_filtered = [v for v in vals if v is not None]
        l_inner = sum(vals_filtered) / len(vals_filtered)


    if l_inner < 0:
        l_inner = 0

    l = l_inner + l_outer


    dG = product.free_energy - reactant.free_energy + dCharge * params['electron_free_energy']
    dG_barrier = l / 4 * (1 + dG / l) ** 2
    reaction['marcus_barrier'] = dG_barrier
    return False

class reactant_and_product_not_isomorphic(MSONable):

    def __init__(self):
        pass

    def __str__(self):
        return "reactants and products are not covalent isomorphic"

    def __call__(self, reaction, mols, params):
        reactant = mols[reaction['reactants'][0]]
        product = mols[reaction['products'][0]]
        if reactant.covalent_hash != product.covalent_hash:
            return True
        else:
            return False


class reaction_default_true(MSONable):

    def __init__(self):
        pass

    def __str__(self):
        return "default true"

    def __call__(self, reaction, mols, params):
        return True

class star_count_diff_above_threshold(MSONable):
    """
    if you want to filter out break-one-form-one reactions, the
    correct value for the threshold is 6.
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __str__(self):
        return "star count diff above threshold=" + str(self.threshold)

    def __call__(self, reaction, mols, params):
        reactant_stars = {}
        product_stars = {}
        tags = set()

        for i in range(reaction['number_of_reactants']):
            reactant_index = reaction['reactants'][i]
            mol = mols[reactant_index]
            for h in mol.star_hashes.values():
                tags.add(h)
                if h in reactant_stars:
                    reactant_stars[h] += 1
                else:
                    reactant_stars[h] = 1

        for j in range(reaction['number_of_products']):
            product_index = reaction['products'][j]
            mol = mols[product_index]
            for h in mol.star_hashes.values():
                tags.add(h)
                if h in product_stars:
                    product_stars[h] += 1
                else:
                    product_stars[h] = 1

        count = 0

        for tag in tags:
            count += abs(reactant_stars.get(tag,0) - product_stars.get(tag,0))

        if count > self.threshold:
            return True
        else:
            return False

class reaction_is_covalent_decomposable(MSONable):
    def __init__(self):
        pass

    def __str__(self):
        return "reaction is covalent decomposable"

    def __call__(self, reaction, mols, params):
        if (reaction['number_of_reactants'] == 2 and
            reaction['number_of_products'] == 2):


            reactant_total_hashes = set()
            for i in range(reaction['number_of_reactants']):
                reactant_id = reaction['reactants'][i]
                reactant = mols[reactant_id]
                reactant_total_hashes.add(reactant.covalent_hash)

            product_total_hashes = set()
            for i in range(reaction['number_of_products']):
                product_id = reaction['products'][i]
                product = mols[product_id]
                product_total_hashes.add(product.covalent_hash)

            if len(reactant_total_hashes.intersection(product_total_hashes)) > 0:
                return True
            else:
                return False

        return False


class metal_coordination_passthrough(MSONable):
    def __init__(self):
        pass

    def __str__(self):
        return "metal coordination passthrough"

    def __call__(self, reaction, mols, params):

        for i in range(reaction['number_of_reactants']):
            reactant_id = reaction['reactants'][i]
            reactant = mols[reactant_id]
            if reactant.formula in m_formulas:
                return True

        for i in range(reaction['number_of_products']):
            product_id = reaction['products'][i]
            product = mols[product_id]
            if product.formula in m_formulas:
                return True

        return False


class fragment_matching_found(MSONable):
    def __init__(self):
        pass

    def __str__(self):
        return "fragment matching found"

    def __call__(self, reaction, mols, params):

        reactant_fragment_indices_list = []
        product_fragment_indices_list = []

        if reaction['number_of_reactants'] == 1:
            reactant = mols[reaction['reactants'][0]]
            for i in range(len(reactant.fragment_data)):
                reactant_fragment_indices_list.append([i])


        if reaction['number_of_reactants'] == 2:
            reactant_0 = mols[reaction['reactants'][0]]
            reactant_1 = mols[reaction['reactants'][1]]
            for i in range(len(reactant_0.fragment_data)):
                for j in range(len(reactant_1.fragment_data)):
                    if (reactant_0.fragment_data[i].number_of_bonds_broken +
                        reactant_1.fragment_data[j].number_of_bonds_broken <= 1):

                        reactant_fragment_indices_list.append([i,j])


        if reaction['number_of_products'] == 1:
            product = mols[reaction['products'][0]]
            for i in range(len(product.fragment_data)):
                product_fragment_indices_list.append([i])


        if reaction['number_of_products'] == 2:
            product_0 = mols[reaction['products'][0]]
            product_1 = mols[reaction['products'][1]]
            for i in range(len(product_0.fragment_data)):
                for j in range(len(product_1.fragment_data)):
                    if (product_0.fragment_data[i].number_of_bonds_broken +
                        product_1.fragment_data[j].number_of_bonds_broken <= 1):

                        product_fragment_indices_list.append([i,j])


        for reactant_fragment_indices in reactant_fragment_indices_list:
            for product_fragment_indices in product_fragment_indices_list:
                reactant_fragment_count = 0
                product_fragment_count = 0
                reactant_bonds_broken = []
                product_bonds_broken = []

                reactant_hashes = dict()
                for reactant_index, frag_complex_index in enumerate(
                        reactant_fragment_indices):

                    fragment_complex = mols[
                        reaction['reactants'][reactant_index]].fragment_data[
                            frag_complex_index]

                    for bond in fragment_complex.bonds_broken:
                        reactant_bonds_broken.append(
                            [(reactant_index, x) for x in bond])

                    for i in range(fragment_complex.number_of_fragments):
                        reactant_fragment_count += 1
                        tag = fragment_complex.fragment_hashes[i]
                        if tag in reactant_hashes:
                            reactant_hashes[tag] += 1
                        else:
                            reactant_hashes[tag] = 1

                product_hashes = dict()
                for product_index, frag_complex_index in enumerate(
                        product_fragment_indices):

                    fragment_complex = mols[
                        reaction['products'][product_index]].fragment_data[
                            frag_complex_index]

                    for bond in fragment_complex.bonds_broken:
                        product_bonds_broken.append(
                            [(product_index, x) for x in bond])


                    for i in range(fragment_complex.number_of_fragments):
                        product_fragment_count += 1
                        tag = fragment_complex.fragment_hashes[i]
                        if tag in product_hashes:
                            product_hashes[tag] += 1
                        else:
                            product_hashes[tag] = 1


                # don't consider fragmentations with both a ring opening and closing
                if (reaction['number_of_reactants'] == 2 and
                    reaction['number_of_products'] == 2 and
                    reactant_fragment_count == 2 and
                    product_fragment_count == 2):
                    continue


                if reactant_hashes == product_hashes:
                    reaction['reactant_bonds_broken'] = reactant_bonds_broken
                    reaction['product_bonds_broken'] = product_bonds_broken
                    reaction['hashes'] = reactant_hashes
                    reaction['reactant_fragment_count'] = reactant_fragment_count
                    reaction['product_fragment_count'] = product_fragment_count

                    return True

        return False


class single_reactant_single_product_not_atom_transfer(MSONable):
    def __init__(self):
        pass

    def __str__(self):
        return "not hydrogen transfer"

    def __call__(self, reaction, mols, params):
        if (reaction['number_of_reactants'] == 1 and
            reaction['number_of_products'] == 1 and
            len(reaction['reactant_bonds_broken']) == 1 and
            len(reaction['product_bonds_broken']) == 1 and
            hydrogen_hash not in reaction['hashes'] and
            fluorine_hash not in reaction['hashes']):

            return True

        return False


class single_reactant_double_product_ring_close(MSONable):
    def __init__(self):
        pass

    def __str__(self):
        return "ring close"


    def __call__(self, reaction, mols, params):

        if (reaction['number_of_reactants'] == 1 and
            reaction['number_of_products'] == 2 and
            len(reaction['reactant_bonds_broken']) == 1 and
            len(reaction['product_bonds_broken']) == 1 and
            reaction['product_fragment_count'] == 2):

            return True

        return False



class concerted_metal_coordination(MSONable):
    def __init__(self):
        pass

    def __str__(self):
        return "concerted metal coordination"

    def __call__(self, reaction, mols, params):

        if (reaction['number_of_reactants'] == 2 and
            reaction['number_of_products'] == 2):

            reactant_0 = mols[reaction['reactants'][0]]
            reactant_1 = mols[reaction['reactants'][1]]
            product_0 = mols[reaction['products'][0]]
            product_1 = mols[reaction['products'][1]]



            if (reactant_0.formula in m_formulas or
                reactant_1.formula in m_formulas or
                product_0.formula in m_formulas or
                product_1.formula in m_formulas):
                return True
            else:
                return False

        return False

class concerted_metal_coordination_one_product(MSONable):
    def __init__(self):
        pass

    def __str__(self):
        return "concerted metal coordination one product"



    def __call__(self, reaction, mols, params):

        if (reaction['number_of_reactants'] == 2 and
            reaction['number_of_products'] == 1):

            reactant_0 = mols[reaction['reactants'][0]]
            reactant_1 = mols[reaction['reactants'][1]]
            product = mols[reaction['products'][0]]

            reactant_covalent_hashes = set([
                reactant_0.covalent_hash,
                reactant_1.covalent_hash])

            if ((reactant_0.formula in m_formulas or
                reactant_1.formula in m_formulas) and
                product.covalent_hash not in reactant_covalent_hashes
                ):
                return True
            else:
                return False

        return False

class concerted_metal_coordination_one_reactant(MSONable):
    def __init__(self):
        pass

    def __str__(self):
        return "concerted metal coordination one reactant"



    def __call__(self, reaction, mols, params):

        if (reaction['number_of_reactants'] == 1 and
            reaction['number_of_products'] == 2):

            product_0 = mols[reaction['products'][0]]
            product_1 = mols[reaction['products'][1]]
            reactant = mols[reaction['reactants'][0]]

            product_covalent_hashes = set([
                product_0.covalent_hash,
                product_1.covalent_hash])

            if ((product_0.formula in m_formulas or
                product_1.formula in m_formulas) and
                reactant.covalent_hash not in product_covalent_hashes
                ):
                return True
            else:
                return False

        return False


class single_reactant_with_ring_break_two(MSONable):
    def __init__(self):
        pass

    def __str__(self):
        return "single reactant with a ring, break two"

    def __call__(self, reaction, mols, params):
        if (reaction["number_of_reactants"] == 1 and
            reaction["number_of_products"] == 2 and
            mols[reaction["reactants"][0]].has_covalent_ring):

            reactant = mols[reaction["reactants"][0]]
            product_1 = mols[reaction["products"][0]]
            product_2 = mols[reaction["products"][1]]
            for fragment_complex in reactant.ring_fragment_data:
                if (set(fragment_complex.fragment_hashes) ==
                    set([product_1.covalent_hash, product_2.covalent_hash])):
                    return True


        return False


class single_product_with_ring_form_two(MSONable):
    def __init__(self):
        pass

    def __str__(self):
        return "single product with a ring, form two"

    def __call__(self, reaction, mols, params):
        if (reaction["number_of_reactants"] == 2 and
            reaction["number_of_products"] == 1 and
            mols[reaction["products"][0]].has_covalent_ring):

            product = mols[reaction["products"][0]]
            reactant_1 = mols[reaction["reactants"][0]]
            reactant_2 = mols[reaction["reactants"][1]]
            for fragment_complex in product.ring_fragment_data:
                if (set(fragment_complex.fragment_hashes) ==
                    set([reactant_1.covalent_hash, reactant_2.covalent_hash])):
                    return True


        return False


# =============================================================================
# molgraph_to_smiles: charge-aware fallback for hexacoordinate phosphorus
#
# OpenBabel's V2000 MOL writer omits the molecular net charge, so anionic
# hexacoordinate-P species (PF6-, fluorophosphate esters, ...) reach RDKit
# looking like NEUTRAL 6-coordinate P, whose maximum allowed valence is 5,
# and fail the sanitizing parse even though the actual anion is legal
# (P- has default valence 6 in RDKit, isoelectronic with S).
#
# The fallback below restores information the molecule really carries: the
# net charge known to pymatgen (mol_graph.molecule.charge) is written onto
# the phosphorus center via a standard "M  CHG" property line and the MOL
# block is re-parsed with FULL sanitization. For F6 P1 with charge -1 this
# yields F[P-](F)(F)(F)(F)F, the correct PF6- SMILES, with the right atom
# count, the right charge, and clean downstream parsing in the atom mapper.
#
# The fallback runs ONLY when the normal sanitizing parse has already
# returned None, so every species that parses normally is untouched, and
# anything the fallback returns has passed RDKit's full valence model with
# the true net charge in place. Species that are genuinely invalid even
# with their charge restored still return None.
# =============================================================================
def _mol_from_molblock_with_net_charge(mol_block, net_charge):
    """Re-parse a V2000 MOL block after restoring the lost net charge.

    OpenBabel's MOL writer drops the molecular charge, so charged
    hexacoordinate-P species arrive at RDKit looking neutral and fail the
    valence check. This helper places the known net charge on the
    most-connected phosphorus atom (tie-break: lowest atom index, fully
    deterministic) via an "M  CHG" property line and re-parses with full
    sanitization.

    Returns an RDKit Mol, or None. Only attempts anything when the net
    charge is nonzero and the atom block contains at least one P.
    """
    if not mol_block or net_charge == 0:
        return None

    lines = mol_block.splitlines()
    if len(lines) < 4:
        return None
    try:
        natoms = int(lines[3][0:3])
        nbonds = int(lines[3][3:6])
    except (ValueError, IndexError):
        return None

    atom_start = 4
    bond_start = atom_start + natoms
    if len(lines) < bond_start + nbonds:
        return None

    # locate phosphorus atoms in the V2000 atom block (symbol field,
    # columns 31-34), recording 1-based indices in file order
    p_indices = []
    for i in range(natoms):
        if lines[atom_start + i][31:34].strip() == "P":
            p_indices.append(i + 1)
    if not p_indices:
        return None

    # connectivity of every atom from the bond block
    degree = {}
    for i in range(nbonds):
        ln = lines[bond_start + i]
        try:
            a = int(ln[0:3])
            b = int(ln[3:6])
        except (ValueError, IndexError):
            continue
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1

    # most-connected P; on ties the lowest atom index wins because
    # p_indices is ascending and the comparison is strictly greater-than
    target = p_indices[0]
    best = degree.get(target, 0)
    for idx in p_indices[1:]:
        d = degree.get(idx, 0)
        if d > best:
            best = d
            target = idx

    # standard CTfile charge property line: "M  CHG" + count + (atom, value)
    chg_line = "M  CHG%3d%4d%4d" % (1, target, net_charge)
    out = []
    inserted = False
    for ln in lines:
        if not inserted and ln.strip() == "M  END":
            out.append(chg_line)
            inserted = True
        out.append(ln)
    if not inserted:
        out.append(chg_line)
        out.append("M  END")

    # full sanitization: whatever comes back has passed RDKit's normal
    # valence model with the true net charge in place
    return Chem.MolFromMolBlock("\n".join(out), sanitize=True)


def molgraph_to_smiles(mol_graph, sanitize: bool = True,
                       remove_hs: bool = False) -> Optional[str]:
    """
    Converts a pymatgen MoleculeGraph to a SMILES string using RDKit.

    Parameters
    ----------
    mol_graph : pymatgen.analysis.graphs.MoleculeGraph
        The molecule graph with bonding info. Only ``.molecule`` is read;
        the edge set is deliberately ignored (see the header comment).

    sanitize : bool
        Whether to sanitize the RDKit molecule during import. Default is True.

    remove_hs : bool
        Whether to remove explicit hydrogens before generating SMILES.
        Default is False.

    Returns
    -------
    smiles : str or None
        The SMILES string if successful, else None.
    """
    try:
        mol_block = mol_graph.molecule.to(fmt="mol")
        if not mol_block:
            # pymatgen 2024.x and later return None from
            # Molecule.to(fmt="mol"); go through OpenBabel directly so
            # the atom mapper still receives a MOL block. Without this
            # every mapping call fails and the AAM filter silently
            # keeps every reaction.
            from pymatgen.io.babel import BabelMolAdaptor
            mol_block = BabelMolAdaptor(mol_graph.molecule).pybel_mol.write("mol")

        rdmol = Chem.MolFromMolBlock(mol_block, sanitize=sanitize)

        if rdmol is None and sanitize:
            # Strict charge-aware fallback: only reachable for species that
            # produce NO SMILES today, so no currently-working species can
            # change. Restores the net charge that OpenBabel's MOL writer
            # dropped (see the header comment) and re-parses with full
            # sanitization.
            rdmol = _mol_from_molblock_with_net_charge(
                mol_block, int(round(mol_graph.molecule.charge)))

        if rdmol is None:
            raise ValueError("RDKit failed to parse MOL block.")

        if remove_hs:
            # both paths went through full sanitization, so the default
            # (sanitizing) RemoveHs is safe here
            rdmol = Chem.RemoveHs(rdmol)

        return Chem.MolToSmiles(rdmol, canonical=True, isomericSmiles=False)

    except Exception as e:
        print(f"[ERROR] SMILES generation failed: {e}")
        return None


class bond_type_change_filter(MSONable):
    """
    Prefilter using MoleculeGraph:
      - Count bonds by element pair (e.g., 'C-O', 'C-C') for reactants/products.
      - Ignore O-Na bonds (treated as coordinate).
      - Discard if total bond-type changes > max_changes (default 2).

    Returns True -> discard; False -> keep.
    """

    def __init__(self, max_changes: int = 2):
        self.max_changes = max_changes
        # sorted element-pair strings treated as coordinate bonds and ignored;
        # Li-O added Jul 2026 so the prefilter agrees with the AAM stage
        # (which exempts both O-Li and O-Na) on Li systems.
        self.ignore_pairs = {"Na-O", "Li-O"}

    def __str__(self):
        return (f"prefilter (MG): discard if >{self.max_changes} "
                f"bond-type changes (ignoring {sorted(self.ignore_pairs)})")

    def __call__(self, reaction, mols, params):
        t0 = time.perf_counter()

        r_pairs = Counter()
        p_pairs = Counter()

        # Reactants
        for i in range(reaction["number_of_reactants"]):
            m = mols[reaction["reactants"][i]]
            mg = m["molecule_graph"] if isinstance(m, dict) else getattr(m, "molecule_graph", getattr(m, "mol_graph"))
            mol = m["molecule"]       if isinstance(m, dict) else getattr(m, "molecule")
            for u, v in mg.graph.edges():
                a1 = _site_symbol(mol[u])
                a2 = _site_symbol(mol[v])
                pair = "-".join(sorted((a1, a2)))
                if pair in self.ignore_pairs:
                    continue
                r_pairs[pair] += 1

        # Products
        for j in range(reaction["number_of_products"]):
            m = mols[reaction["products"][j]]
            mg = m["molecule_graph"] if isinstance(m, dict) else getattr(m, "molecule_graph", getattr(m, "mol_graph"))
            mol = m["molecule"]       if isinstance(m, dict) else getattr(m, "molecule")
            for u, v in mg.graph.edges():
                a1 = _site_symbol(mol[u])
                a2 = _site_symbol(mol[v])
                pair = "-".join(sorted((a1, a2)))
                if pair in self.ignore_pairs:
                    continue
                p_pairs[pair] += 1

        # Total bond-type changes (element-pair level, ignoring O-Na)
        diff = (r_pairs - p_pairs) + (p_pairs - r_pairs)
        total_changes = sum(diff.values())

        elapsed = time.perf_counter() - t0
        #print(f"[TIMER][bond_type_change_filter_mg] total={elapsed:.4f}s  changes={total_changes}  (ignore {self.ignore_pairs})")

        if total_changes > self.max_changes:
            return True
        return False



default_reaction_decision_tree = [

    (metal_metal_reaction(), Terminal.DISCARD),
    # redox branch
    (is_redox_reaction(), [

        (too_many_reactants_or_products(), Terminal.DISCARD),
        (dcharge_too_large(), Terminal.DISCARD),
        (reactant_and_product_not_isomorphic(), Terminal.DISCARD),
        (dG_above_threshold(0.0, "free_energy", 0.0), Terminal.DISCARD),
        (reaction_default_true(), Terminal.KEEP)
    ]),

    (dG_above_threshold(0.0, "solvation_free_energy", 0.0), Terminal.DISCARD),


    # (single_reactant_with_ring_break_two(), Terminal.KEEP),
    # (single_product_with_ring_form_two(), Terminal.KEEP),

    (star_count_diff_above_threshold(6), Terminal.DISCARD),

    (reaction_is_covalent_decomposable(), Terminal.DISCARD),

    (concerted_metal_coordination(), Terminal.DISCARD),

    (concerted_metal_coordination_one_product(), Terminal.DISCARD),

    (concerted_metal_coordination_one_reactant(), Terminal.DISCARD),

    (metal_coordination_passthrough(), Terminal.KEEP),

    (bond_type_change_filter(), Terminal.DISCARD),

    (fragment_matching_found(), [
        (single_reactant_single_product_not_atom_transfer(), Terminal.DISCARD),
        (single_reactant_double_product_ring_close(), Terminal.DISCARD),
        (reaction_default_true(), Terminal.KEEP)]
    ),
    
    #(bond_type_change_filter(), Terminal.DISCARD),

    #(bond_change_filter(), Terminal.DISCARD),

    (reaction_default_true(), Terminal.DISCARD)
    ]
