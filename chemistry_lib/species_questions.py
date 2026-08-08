"""Species decision tree: the questions applied to each molecule during
species filtering. Adapted from the framework of Barter et al. with this
project's modifications."""

from chemistry_lib.mol_entry import MoleculeEntry, FragmentComplex
import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
import copy
from functools import partial
from chemistry_lib.constants import li_ec, Na_ec, Terminal, m_formulas, metals
import numpy as np
from monty.json import MSONable
from itertools import combinations
from ase import Atoms
from ase.io import write

"""
species decision tree:

A question is a function q(mol_entry) -> Bool

Unlike for reaction filtering, these questions should not modify the mol_entry in any way.

A node is either a Terminal or a non empty list [(question, node)]

class Terminal(Enum):
    KEEP = 1
    DISCARD = -1

For the return value of a question, True means travel to this node and False means try next question in the list.

for non terminal nodes, it is an error if every question returns False. i.e getting stuck at a non terminal node is an error.

Once a Terminal node is reached, it tells us whether to keep or discard the species.
"""

def run_decision_tree(mol_entry,
                      decision_tree,
                      decision_pathway=None):

    node = decision_tree

    while type(node) == list:
        next_node = None
        for (question, new_node) in node:
            if question(mol_entry):

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
        raise Exception("unexpected node type reached")


class metal_ion_filter(MSONable):
    "only allow positively charged metal ions"
    def __init__(self):
        pass

    def __call__(self, mol_entry):
        if mol_entry.formula in m_formulas and mol_entry.charge <= 0:
            return True
        else:
            return False

class mol_not_connected(MSONable):
    def __init__(self):
        pass

    def __call__(self, mol):
        #if not nx.is_connected(mol.graph):
        #    print("Not connected")
        return not nx.is_connected(mol.graph)

class spin_multiplicity_filter(MSONable):
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, mol):
        # Datasets may carry null partial_spins for some species (monoatomics
        # in particular), so guard before indexing. The penalty needs spin
        # density on two or more atoms, which a single atom cannot have.
        if (mol.spin_multiplicity == 2
                and mol.partial_spins_nbo is not None):
            num_partial_spins_above_threshold = 0
            for i in range(mol.num_atoms):
                if mol.partial_spins_nbo[i] > self.threshold:
                    num_partial_spins_above_threshold += 1

            if num_partial_spins_above_threshold >= 2:
                mol.penalty += 1

        return False

class positive_penalty(MSONable):
    def __init__(self):
        pass

    def __call__(self, mol):
        if mol.penalty > 0:
            return True
        else:
            return False

class add_star_hashes(MSONable):
    def __init__(self):
        pass

    def __call__(self, mol):
        for i in range(mol.num_atoms):
            if i not in mol.m_inds:
                neighborhood = nx.generators.ego.ego_graph(
                    mol.covalent_graph,
                    i,
                    1,
                    undirected=True) #.to_undirected()
                #print("--------------------------------------------------------------------------------Is it directed graph:",neighborhood.is_directed())
                mol.star_hashes[i] = weisfeiler_lehman_graph_hash(
                    neighborhood,
                    node_attr='specie')

        return False

class add_unbroken_fragment(MSONable):
    def __init__(self):
        pass

    def __call__(self, mol):
        if mol.formula in m_formulas:
            return False

        fragment_complex = FragmentComplex(
             1,
             0,
             [],
             [mol.covalent_hash])

        mol.fragment_data.append(fragment_complex)

        return False

class add_single_bond_fragments(MSONable):

    def __init__(self):
        pass

    def __call__(self, mol):

        if mol.formula in m_formulas:
            return False



        for edge in mol.covalent_graph.edges:
            fragments = []
            h = copy.deepcopy(mol.covalent_graph)
            h.remove_edge(*edge)
            connected_components = nx.algorithms.components.connected_components(h)
            for c in connected_components:

                subgraph = h.subgraph(c)

                fragment_hash = weisfeiler_lehman_graph_hash(
                    subgraph,
                    node_attr='specie')


                fragments.append(fragment_hash)

            fragment_complex = FragmentComplex(
                len(fragments),
                1,
                [edge[0:2]],
                fragments)

            mol.fragment_data.append(fragment_complex)

        return False

class has_covalent_ring(MSONable):
    def __init__(self):
        pass

    def __call__(self, mol):
        # if mol is a metal, mol.covalent_graph is empty
        if mol.formula in m_formulas:
            mol.has_covalent_ring = False
        else:
            mol.has_covalent_ring = not nx.is_tree(mol.covalent_graph)

        if mol.has_covalent_ring:
            mol.ring_fragment_data = []

        return mol.has_covalent_ring


class covalent_ring_fragments(MSONable):
    def __init__(self):
        pass

    def __call__(self, mol):
        # maps edge to graph with that edge removed
        ring_edges = {}

        for edge in mol.covalent_graph.edges:
            h = copy.deepcopy(mol.covalent_graph)
            h.remove_edge(*edge)
            if nx.is_connected(h):
                ring_edges[edge] = {
                    'modified_graph' : h,
                    'node_set' : set([edge[0],edge[1]])
                }


        for ring_edge_1, ring_edge_2 in combinations(ring_edges,2):

            if ring_edges[ring_edge_1]['node_set'].isdisjoint(
                    ring_edges[ring_edge_2]['node_set']):


                potential_edges =  [ (ring_edge_1[0], ring_edge_2[0],0),
                                     (ring_edge_1[0], ring_edge_2[1],0),
                                     (ring_edge_1[1], ring_edge_2[0],0),
                                     (ring_edge_1[1], ring_edge_2[1],0) ]

                one_bond_away = False
                for ring_edge_3 in ring_edges:
                    if ring_edge_3 in potential_edges:
                        one_bond_away = True

                if one_bond_away:
                    h = copy.deepcopy(ring_edges[ring_edge_1]['modified_graph'])
                    h.remove_edge(*ring_edge_2)
                    if nx.is_connected(h):
                        continue
                    else:
                        fragments = []
                        connected_components = nx.algorithms.components.connected_components(h)
                        for c in connected_components:

                            subgraph = h.subgraph(c)

                            fragment_hash = weisfeiler_lehman_graph_hash(
                                subgraph,
                                node_attr='specie')


                            fragments.append(fragment_hash)

                        fragment_complex = FragmentComplex(
                            len(fragments),
                            2,
                            [ring_edge_1[0:2], ring_edge_2[0:2]],
                            fragments)

                        mol.ring_fragment_data.append(fragment_complex)

        return False


class metal_complex(MSONable):
    def __init__(self):
        pass

    def __call__(self, mol):
        # if mol is a metal, it isn't a metal complex
        if mol.formula in m_formulas:
            return False

        return not nx.is_connected(mol.covalent_graph)


class fix_hydrogen_bonding(MSONable):
    def __init__(self):
        pass

    def __call__(self, mol):
        if mol.num_atoms > 1:
            for i in range(mol.num_atoms):
                if mol.species[i] == 'H':

                    adjacent_atoms = []

                    for bond in mol.graph.edges:
                        if i in bond[0:2]:

                            if i == bond[0]:
                                adjacent_atom = bond[1]
                            else:
                                adjacent_atom = bond[0]

                            displacement = (mol.atom_locations[adjacent_atom] -
                                            mol.atom_locations[i])

                            dist = np.inner(displacement, displacement)

                            adjacent_atoms.append((adjacent_atom, dist))

                    if adjacent_atoms:
                        closest_atom, _ = min(adjacent_atoms, key=lambda pair: pair[1])

                        for adjacent_atom, _ in adjacent_atoms:
                            if adjacent_atom != closest_atom:
                                mol.graph.remove_edge(i, adjacent_atom)
                                if adjacent_atom in mol.covalent_graph:
                                    mol.covalent_graph.remove_edge(i, adjacent_atom)
                    else:
                        print(f"Warning: No adjacent atoms found for hydrogen at index {i} in molecule {mol.entry_id}")
                        # Create an ASE Atoms object from mol
                        atoms = Atoms(symbols=mol.species, positions=mol.atom_locations)

                        # Save the molecule as an XYZ file
                        filename = f"hyd_mol_{mol.entry_id}.xyz"
                        write(filename, atoms)
                        print(f"Molecule saved as {filename}")

        return False


class bad_metal_coordination(MSONable):
    def __init__(self):
        pass

    def __call__(self, mol):

        if mol.formula not in m_formulas:

            if (len(metals.intersection(set(mol.species))) > 0 and
                mol.number_of_coordination_bonds == 0):
                print("Bad metal coordination filtered out:", mol.entry_id)
                print("\n")
                return True

        return False


class set_solvation_free_energy(MSONable):
    """
    metal atoms coordinate with the surrounding solvent. We need to correct
    free energy to take this into account. The correction is
    solvation_correction * (
           max_coodination_bonds -
           number_of_coordination_bonds_in_mol).
    Since coordination bonding can't reliably be detected from the molecule
    graph, we search for all atoms within a radius of the metal atom and
    discard them if they are positively charged.
    """

    def __init__(self, solvation_env):
        self.solvation_env = solvation_env

    def __call__(self, mol):
        correction = 0.0
        mol.number_of_coordination_bonds = 0

        for i in mol.m_inds:

            species = mol.species[i]
            partial_charge = mol.partial_charges_nbo[i]

            if partial_charge < 1.2:
                effective_charge = "_1"
            elif partial_charge >= 1.2:
                effective_charge = "_2"

            coordination_partners = list()
            species_charge = species + effective_charge
            #print(self.solvation_env)
            radius = self.solvation_env["coordination_radius"][species_charge]
            #print(species,"   ",partial_charge,"   ",species_charge)
            for j in range(mol.num_atoms):
                if j != i:
                    displacement_vector = (
                        mol.atom_locations[j] -
                        mol.atom_locations[i])
                    if (np.inner(displacement_vector, displacement_vector)
                        < radius ** 2 and (
                            mol.partial_charges_resp[j] < 0 or
                            mol.partial_charges_mulliken[j] < 0 or
                            mol.partial_charges_nbo[j] < 0)):
                        if not mol.graph.has_edge(i,j):
                            mol.graph.add_edge(i,j)
                        coordination_partners.append(j)

            number_of_coordination_bonds = len(coordination_partners)
            mol.number_of_coordination_bonds += number_of_coordination_bonds
            correction += self.solvation_env[
                "solvation_correction"][species_charge] * (
                self.solvation_env[
                    "max_number_of_coordination_bonds"][species_charge] -
                number_of_coordination_bonds)
            #print("Calculated correction: ",correction)
        #print(mol)
        if mol.free_energy is None:
            # Missing DFT free-energy in the source dataset.  Earlier code
            # silently stamped a magic -162.1192 eV here, which turns a
            # data error into a wrong-physics outcome.  We raise instead
            # so the bad input is exposed loudly; clean the source JSON
            # (or pre-filter it) before re-running.
            entry_id = getattr(mol, "entry_id", "<unknown>")
            formula = getattr(mol, "formula", "<unknown>")
            raise ValueError(
                f"species_filter: mol.free_energy is None for entry "
                f"{entry_id} (formula={formula}). The source dataset is "
                "missing a Gibbs free energy for this species; either "
                "supply it or drop the entry before running species_filter."
            )
        mol.solvation_free_energy = correction + mol.free_energy
        return False


class species_default_true(MSONable):
    def __init__(self):
        pass

    def __call__(self, mol):
        return True


def compute_graph_hashes(mol):
    mol.total_hash = weisfeiler_lehman_graph_hash(
        mol.graph,
        node_attr='specie')

    mol.covalent_hash = weisfeiler_lehman_graph_hash(
        mol.covalent_graph,
        node_attr='specie')

    return False


class metal_only_coordinated_to_carbon(MSONable):
    """
    Discard species where a metal is bonded/coordinated to a C atom
    and has no O or F coordination partners.

    Returns True  -> species should be discarded
    Returns False -> species is okay
    """
    def __init__(self):
        pass

    def __call__(self, mol):
        # Skip isolated metal ions like Li+, Na+, etc.
        if mol.formula in m_formulas:
            return False

        for m_ind in mol.m_inds:
            if mol.species[m_ind] not in metals:
                continue

            neighbors = []
            for edge in mol.graph.edges:
                if m_ind in edge[0:2]:
                    neighbor = edge[1] if edge[0] == m_ind else edge[0]
                    neighbors.append(neighbor)

            if not neighbors:
                continue

            neighbor_species = [mol.species[j] for j in neighbors]

            has_carbon = "C" in neighbor_species
            has_oxygen_or_fluorine = ("O" in neighbor_species) or ("F" in neighbor_species)

            if has_carbon and not has_oxygen_or_fluorine:
                print(f"Filtered out metal-carbon-only coordination: {mol.entry_id}")
                return True

        return False


class metal_only_coordinated_to_carbon(MSONable):
    """Discard species where a metal is bonded/coordinated to a C atom
    and has no O or F coordination partners. Such species are usually
    optimization artifacts rather than solution-phase chemistry.
    """
    def __init__(self):
        pass

    def __call__(self, mol):
        if mol.formula in m_formulas:
            return False

        for m_ind in mol.m_inds:
            if mol.species[m_ind] not in metals:
                continue

            neighbors = []
            for edge in mol.graph.edges:
                if m_ind in edge[0:2]:
                    neighbor = edge[1] if edge[0] == m_ind else edge[0]
                    neighbors.append(neighbor)

            if not neighbors:
                continue

            neighbor_species = [mol.species[j] for j in neighbors]

            has_carbon = "C" in neighbor_species
            has_oxygen_or_fluorine = ("O" in neighbor_species) or ("F" in neighbor_species)

            if has_carbon and not has_oxygen_or_fluorine:
                print(f"Filtered out metal-carbon-only coordination: {mol.entry_id}")
                return True

        return False


class neutral_metal_filter(MSONable):
    def __init__(self, cutoff):
        self.cutoff = cutoff

    def __call__(self, mol):
        for i in mol.m_inds:
            if (mol.species[i] in metals and
                mol.partial_charges_nbo[i] < self.cutoff):
                print("Molecule excluded due to neutral metal:", mol.entry_id)
                return True

        return False

class charge_too_big(MSONable):
    def __init__(self):
        pass

    def __call__(self, mol):
        if mol.charge > 1 or mol.charge < -1:
            return True

        else:
            return False

# any species filter which modifies bonding has to come before
# any filter checking for connectivity (which includes the metal-centric complex filter)


def make_species_decision_tree(solvation_environment,
                               neutral_metal_cutoff=None,
                               drop_metal_carbon_only=None):
    """Build the species decision tree against a given solvation environment
    (one of the dicts in chemistry_lib.constants, e.g. li_ec, Na_ec).

    The factory exposes the solvation environment as a parameter so the
    same tree shape works for either supported cation chemistry.

    The two filter parameters default per cation, detected from the
    solvation environment. Na chemistry needs a much smaller
    neutral_metal_cutoff because NBO places the unpaired electron of
    vertically reduced Na complexes on the metal itself (Na charge is
    about +0.04 in ring-intact NaEC0), so the stock 0.1 cutoff would
    discard the species that carry the reduction cascade. Li complexes
    keep the electron on the ligand (Li stays near +0.9), so the stock
    cutoff is safe there. Na also enables the metal-carbon-only artifact
    filter.
    """
    is_na = any(k.startswith("Na")
                for k in solvation_environment["solvation_correction"])
    if neutral_metal_cutoff is None:
        neutral_metal_cutoff = 0.001 if is_na else 0.1
    if drop_metal_carbon_only is None:
        drop_metal_carbon_only = is_na
    tree = [
        (fix_hydrogen_bonding(), Terminal.KEEP),
        (set_solvation_free_energy(solvation_environment), Terminal.KEEP),
    ]
    if drop_metal_carbon_only:
        tree.append((metal_only_coordinated_to_carbon(), Terminal.DISCARD))
    tree += [
        (charge_too_big(), Terminal.DISCARD),
        (neutral_metal_filter(neutral_metal_cutoff), Terminal.DISCARD),
        (compute_graph_hashes, Terminal.KEEP),
        (metal_ion_filter(), Terminal.DISCARD),
        (bad_metal_coordination(), Terminal.DISCARD),
        (mol_not_connected(), Terminal.DISCARD),
        (metal_complex(), Terminal.DISCARD),
        (spin_multiplicity_filter(0.4), Terminal.DISCARD),
        (add_star_hashes(), Terminal.KEEP),
        (add_unbroken_fragment(), Terminal.KEEP),
        (add_single_bond_fragments(), Terminal.KEEP),
        (species_default_true(), Terminal.KEEP),
    ]
    return tree


# Convenience pre-built trees for the two supported chemistries. Filter
# parameters resolve per cation inside the factory.
na_species_decision_tree = make_species_decision_tree(Na_ec)
li_species_decision_tree = make_species_decision_tree(li_ec)

