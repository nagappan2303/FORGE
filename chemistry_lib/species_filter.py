"""
Phase 1: species filtering
input: a list of dataset entries
output: a filtered list of mol_entries with fixed indices
description: this is where we remove isomorphic species, and do other forms of filtering. Species decision tree is what we use for filtering.

species isomorphism filtering:

The input dataset entries will often contain isomorphic molecules. Identifying such isomorphisms doesn't fit into the species decision tree, so we have it as a preprocessing phase.

Adapted from the framework of Barter et al. with this project's modifications.
"""
from chemistry_lib.mol_entry import MoleculeEntry
import pickle
from chemistry_lib.species_questions import run_decision_tree
from chemistry_lib.constants import Terminal
from chemistry_lib._logging import log_message
import networkx as nx
import networkx.algorithms.isomorphism as iso
from chemistry_lib.report_generator import ReportGenerator
import re

def sort_into_tags(mols):
    isomorphism_buckets = {}
    for mol in mols:

        tag = (mol.charge, mol.formula, mol.covalent_hash)

        if tag in isomorphism_buckets:
            isomorphism_buckets[tag].append(mol)
        else:
            isomorphism_buckets[tag] = [mol]

    return isomorphism_buckets


def really_covalent_isomorphic(mol1, mol2):
    """
    check for isomorphism directly instead of using hash.
    warning: this is really slow. It is used in species filtering
    to avoid hash collisions. Do not use it anywhere else.
    """
    return nx.is_isomorphic(
        mol1.covalent_graph,
        mol2.covalent_graph,
        node_match = iso.categorical_node_match('specie', None)
    )



def groupby(equivalence_relation, xs):
    """
    warning: this has slightly different semantics than
    itertools groupby which depends on ordering.
    """
    groups = []

    for x in xs:
        group_found = False
        for group in groups:
            if equivalence_relation(x, group[0]):
                group.append(x)
                group_found = True
                break

        if not group_found:
            groups.append([x])

    return groups


def species_filter(
        dataset_entries,
        mol_entries_pickle_location,
        species_report,
        species_decision_tree,
        coordimer_weight,
        species_logging_decision_tree=Terminal.DISCARD,
        generate_unfiltered_mol_pictures=False
):

    """
    run each molecule through the species decision tree and then choose the lowest weight
    coordimer based on the coordimer_weight function.
    """

    log_message("starting species filter")
    log_message("loading molecule entries from json")
    
    mol_entries_unfiltered = [
        MoleculeEntry.from_dataset_entry(e) for e in dataset_entries ]
    print("\nNo.of molecules before begining species filration: ",len(mol_entries_unfiltered))
    print("\n")

    #for i in range(0,len(mol_entries_unfiltered)):
        #print(mol_entries_unfiltered[i])
    log_message("generating unfiltered mol pictures")

    report_generator = ReportGenerator(
        mol_entries_unfiltered,
        species_report,
        mol_pictures_folder_name='mol_pictures_unfiltered',
        rebuild_mol_pictures=generate_unfiltered_mol_pictures
    )

    report_generator.emit_text("species report")

    log_message("applying local filters")
    mol_entries_filtered = []
    filter_counts = {str(filter_func): 0 for filter_func, _ in species_decision_tree}
    cc=0
    # note: it is important here that we are applying the local filters before
    # the non local ones. We remove some molecules which are lower energy
    # than other more realistic lithomers.

    for i, mol in enumerate(mol_entries_unfiltered):
        log_message("filtering " + mol.entry_id)
        decision_pathway = []
        if run_decision_tree(mol, species_decision_tree, decision_pathway):
            mol_entries_filtered.append(mol)
            cc=cc+1
        else:
            for item in decision_pathway:
                if isinstance(item, tuple):  # Ensure it's unpackable
                    filter_func, terminal = item
                    if terminal == Terminal.DISCARD:
                        filter_counts[str(filter_func)] += 1
                        break
                else:
                    # Assume item itself is a filter function object
                    filter_counts[str(item)] += 1
                    break        
        if run_decision_tree(mol, species_logging_decision_tree):
            report_generator.emit_verbatim(
                '\n'.join([str(f) for f in decision_pathway]))

            report_generator.emit_text("number: " + str(i))
            report_generator.emit_text("entry id: " + mol.entry_id)
            report_generator.emit_text("uncorrected free energy: " +
                                       str(mol.free_energy))

            report_generator.emit_text(
                "number of coordination bonds: " +
                str(mol.number_of_coordination_bonds))

            report_generator.emit_text(
                "corrected free energy: " +
                str(mol.solvation_free_energy))

            report_generator.emit_text(
                "formula: " + mol.formula)

            report_generator.emit_molecule(i, include_index=False)
            report_generator.emit_newline()


    report_generator.finished()
    print("\nNo.of molecules after local filters",cc)
    print("\nFilter exclusion counts:")
    for filter_func, count in filter_counts.items():
        if(filter_func.split(' ')[0]=="<function"):
            name=filter_func.split(' ')[1]
        else:
            name=filter_func.split('.')[2].split(' ')[0]
        print(f"{name}: {count} molecules excluded")
        
    # python doesn't have shared memory. That means that every worker during
    # reaction filtering must maintain its own copy of the molecules.
    # for this reason, it is good to remove attributes that are only used
    # during species filtering.
    log_message("clearing unneeded attributes")
    for m in mol_entries_filtered:
        del m.partial_charges_resp
        del m.partial_charges_mulliken
        del m.partial_charges_nbo
        del m.partial_spins_nbo
        del m.atom_locations

    # currently, take lowest energy mol in each iso class
    log_message("applying non local filters")
    non_local_filter_count = 0

    def collapse_isomorphism_group(g):
        lowest_energy_coordimer = min(g,key=coordimer_weight)
        return lowest_energy_coordimer


    mol_entries = []

    for tag_group in sort_into_tags(mol_entries_filtered).values():
        for iso_group in groupby(really_covalent_isomorphic, tag_group):
            iso_group_list = list(iso_group)
            selected_mol = collapse_isomorphism_group(iso_group)
            if selected_mol:
                mol_entries.append(selected_mol)
                non_local_filter_count = non_local_filter_count + (len(iso_group_list) - 1)
                removed_mols = [
                    m for m in iso_group_list
                    if m.entry_id != selected_mol.entry_id
                ]
                if removed_mols:
                    print("\nNon-local filter group:")
                    #print(f"  tag = {tag}")
                    print(f"  kept    : {selected_mol.entry_id}")
                    print("  removed : " + ", ".join(m.entry_id for m in removed_mols))

                    # optional: also print weights/energies
                    print("  details:")
                    for m in iso_group_list:
                        try:
                            w = coordimer_weight(m)
                        except Exception:
                            w = "N/A"

                        print(
                        f"    {m.entry_id} | weight = {w} | "
                        f"solv_G = {getattr(m, 'solvation_free_energy', 'N/A')} | "
                        f"free_G = {getattr(m, 'free_energy', 'N/A')} | "
                        f"penalty = {getattr(m, 'penalty', 'N/A')}"
                        )

                #print("\nHiii")
            else:
                non_local_filter_count = non_local_filter_count + len(iso_group)

    print(f"\nNon-local filters excluded {non_local_filter_count} molecules")
            
            #mol_entries.append(
            #    collapse_isomorphism_group(iso_group))


    log_message("assigning indices")

    for i, e in enumerate(mol_entries):
        e.ind = i


    log_message("creating molecule entry pickle")
    # ideally we would serialize mol_entries to a json
    # some of the auxilary_data we compute
    # has frozen set keys, so doesn't seralize well into json format.
    # pickles work better in this setting
    with open(mol_entries_pickle_location, 'wb') as f:
        pickle.dump(mol_entries, f)

    log_message("species filtering finished. " +
                str(len(mol_entries)) +
                " species")

    return mol_entries
