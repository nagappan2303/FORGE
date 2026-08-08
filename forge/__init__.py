"""
FORGE: seeded, flux-driven iterative chemical reaction network construction.

Wraps the chemistry_lib species_filter + reaction-decision-tree machinery in
an outer loop where reactions are generated only for species connected to a
growing "core" set, and species are promoted into the core only after they
carry measurable flux in a short kMC run.
"""

__version__ = "1.0.0"
