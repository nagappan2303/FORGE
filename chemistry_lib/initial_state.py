"""Builds the initial_state.sqlite consumed by GMC: species counts, an empty
trajectories table, and kMC factors. Adapted from the framework of Barter
et al. with this project's modifications."""

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.analysis.fragmenter import metal_edge_extender
import sqlite3


def find_mol_entry_from_xyz_and_charge(mol_entries, xyz_file_path, charge,
                                       spin_multiplicity=None):
    """Find the mol_entry whose molecule graph matches the structure in the
    given xyz file at the given charge. When spin_multiplicity is given it
    must match as well. Returns the entry index, or None if nothing matches.
    """
    target_mol_graph = MoleculeGraph.with_local_env_strategy(
        Molecule.from_file(xyz_file_path), OpenBabelNN()
    )

    # correction to the molecule graph
    target_mol_graph = metal_edge_extender(target_mol_graph)
    print("Searching for\n")
    print(target_mol_graph)
    for mol_entry in mol_entries:
        if mol_entry.charge != charge:
            continue
        if (spin_multiplicity is not None
                and mol_entry.spin_multiplicity != spin_multiplicity):
            continue
        if target_mol_graph.isomorphic_to(mol_entry.mol_graph):
            print("Matched ind =", mol_entry.ind,
                  "entry_id =", mol_entry.entry_id)
            return mol_entry.ind
    return None

def find_mol_entry_by_entry_id(mol_entries, entry_id):
    """
    given an entry_id, return the corresponding mol entry index
    """

    for m in mol_entries:
        if m.entry_id == entry_id:
            return m.ind

create_initial_state_table = """
    CREATE TABLE initial_state (
            species_id             INTEGER NOT NULL PRIMARY KEY,
            count                  INTEGER NOT NULL
    );
"""

create_trajectories_table = """
    CREATE TABLE trajectories (
            seed         INTEGER NOT NULL,
            step         INTEGER NOT NULL,
            reaction_id  INTEGER NOT NULL,
            time         REAL NOT NULL
    );
"""

create_factors_table = """
    CREATE TABLE factors (
            factor_zero      REAL NOT NULL,
            factor_two       REAL NOT NULL,
            factor_duplicate REAL NOT NULL
    );
"""

# Newer GMC builds (RNMC with checkpoint support) expect these two tables
# in the initial-state database even when checkpointing is disabled.
# Older builds ignore them, so they are always created.
create_interrupt_state_table = """
    CREATE TABLE IF NOT EXISTS interrupt_state (
            seed        INTEGER NOT NULL,
            species_id  INTEGER NOT NULL,
            count       INTEGER NOT NULL
    );
"""

create_interrupt_cutoff_table = """
    CREATE TABLE IF NOT EXISTS interrupt_cutoff (
            seed        INTEGER NOT NULL,
            step        INTEGER NOT NULL,
            time        REAL NOT NULL
    );
"""


def insert_initial_state(
        initial_state,
        mol_entries,
        initial_state_db,
        factor_zero = 1.0,
        factor_two = 1.0,
        factor_duplicate = 0.5
):
    """
    initial state is a dict mapping species ids to counts.
    """

    rn_con = sqlite3.connect(initial_state_db)
    rn_cur = rn_con.cursor()
    rn_cur.execute(create_initial_state_table)
    rn_cur.execute(create_trajectories_table)
    rn_cur.execute(create_factors_table)
    rn_cur.execute(create_interrupt_state_table)
    rn_cur.execute(create_interrupt_cutoff_table)
    rn_con.commit()

    rn_cur.execute(
        "INSERT INTO factors VALUES (?,?,?)",
        (factor_zero, factor_two, factor_duplicate))

    num_species = len(mol_entries)


    for i in range(num_species):
        rn_cur.execute(
            "INSERT INTO initial_state VALUES (?,?)",
            (i, initial_state.get(i,0)))

    rn_con.commit()



