# coding: utf-8
# Copyright (c) MR.Net development team


from enum import Enum
from monty.json import MSONable

# Basic constants

# Room temperature (25 C) in Kelvin
ROOM_TEMP = 298.15

# Boltzmann constant in eV / K
KB = 8.617333262 * 10 ** -5

# Planck constant in eV * s
PLANCK = 4.135667696 * 10 ** -15

class Terminal(MSONable, Enum):
    KEEP = 1
    DISCARD = -1

metals = frozenset(["Li", "Na"])
m_formulas = frozenset([m + "1" for m in metals])


# solvation environments. FORGE currently supports Li and Na cation
# chemistries. Add new entries here (and a matching tree in
# species_questions.py) for other cations.
li_ec = {
    "solvation_correction" : {
        "Li_1" : -0.68
    },

    "coordination_radius" : {
        "Li_1" : 2.4
    },

    "max_number_of_coordination_bonds" : {
        "Li_1" : 4
    }
}

Na_ec = {
    "solvation_correction" : {
        "Na_1" : -0.44
    },

    "coordination_radius" : {
        "Na_1" : 2.4
    },

    "max_number_of_coordination_bonds" : {
        "Na_1" : 5
    }
}
