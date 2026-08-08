# `aam_src/`: bundled atom-atom mapper

This is a self-contained copy of the AAM (atom-atom mapping) mapper used
by FORGE's inline elementary-step filter (`forge.aam_filter`).

The Ising formulation and annealing core follow the atom-mapping method of
Ali et al. This project adds: explicit hydrogen atoms as graph nodes, a
direct enumeration fast path for reactions with an empty heavy-atom bond
graph on either side, subprocess timeout isolation (in `forge.aam_filter`),
and a canonical reaction-SMILES cache (`forge.aam_cache`).

## Files
- `reaction_mapper.py`: public API consumed by `forge.aam_filter`. The
  call site uses `analyze_reaction(rxn_str, timeout_s)`.
- `MapIsing.py`: the Ising-formulation atom-atom mapper.
- `optim_wrapper.py`, `optim/`: the Ising solver (uses `dwave-NEAL` via
  `optim/ising/samplers.py::NealSampler`).

## How it gets imported

`aam_src/__init__.py` inserts the package directory at the front of
`sys.path` so the bare imports inside `MapIsing.py`,
`optim_wrapper.py`, and `reaction_mapper.py` (e.g. `import MapIsing as
mi`, `from optim.core.problem import *`) resolve when FORGE does
`from aam_src.reaction_mapper import analyze_reaction`.

## Required packages

`rdkit`, `networkx`, `pandas`, `scipy`, `sympy`, `dwave-neal`. All are
declared in the top-level `pyproject.toml`.
