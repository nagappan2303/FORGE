# FORGE

FORGE (Flux-Orchestrated Reaction Generation Engine) builds chemical reaction
networks for electrochemical systems iteratively rather than by exhaustive
enumeration. Two ideas carry the method. First, flux-guided growth: starting
from a handful of seed species, each iteration enumerates only reactions that
touch the current core species (seeded bucketing against an on-disk
composition index), runs a short screening kMC, and promotes into the core any
species whose share of total produced flux clears the normalized flux
threshold epsilon. Second, inline elementary-step enforcement: every surviving
reaction is atom-to-atom mapped by an Ising-formulation solver; the mapping
yields the bond edits n_b/n_f, and reactions that violate the elementary-step
criterion are rejected before they ever enter the network.

## Repository layout

| Directory | Contents |
|---|---|
| `forge/` | The iteration loop: config, seeded bucketing, composition-index reaction generation, screening kMC via GMC, flux extraction and promotion, inline AAM filter and cache. |
| `aam_src/` | Bundled Ising atom-to-atom mapper (adapted from Ali et al.), consumed by `forge.aam_filter`. |
| `chemistry_lib/` | Species and reaction decision trees plus kMC analysis and report generation, adapted from the framework of Barter et al. |
| `tools/` | Post-run utilities: `production_kmc.py`, `make_sink_report.py`, `run_status.py`, `find_seed_indices.py`. |
| `examples/` | One self-contained Li+/EC example (`run_li_ec_demo.py`) with its bundled 608-species test pool (`data/li_ec_test.pickle`). |

## Requirements and installation

Python >= 3.8.

```bash
git clone https://github.com/nagappan2303/FORGE.git
cd FORGE
pip install -e .
```

The optional network-rendering utilities need pycairo
(`pip install -e ".[viz]"`); everything else, including all reports, works
without it.
`mpi4py` is needed only for the multi-node enumeration backend
(`enumeration_backend = "mpi"`), typically launched through your cluster's
batch system.

Two external programs:

- **GMC (required).** FORGE contains no kMC implementation of its own; all
  kinetic Monte Carlo is delegated to the compiled GMC engine from RNMC
  (https://github.com/BlauGroup/RNMC). Build it once:

  ```bash
  git clone https://github.com/BlauGroup/RNMC
  cd RNMC/GMC && mkdir -p ../build && make GMC
  ```

  The build needs a C++17 compiler, GSL, and SQLite3 (on macOS:
  `brew install gsl`; on most Linux systems: the `gsl` and `sqlite`
  development packages). The binary lands at `RNMC/build/GMC`; pass it to
  the example with `--gmc`, or set `gmc_binary_path` in your config. FORGE
  raises at startup if the binary is missing or not executable.
- **pdflatex (optional).** End-of-run reports are written as `.tex` and
  compiled to PDF when `pdflatex` is available; otherwise the `.tex` files
  are left as-is.

## Running the example

A complete FORGE run on a bundled Li+/EC test pool (608 C/H/O/Li species
drawn from the production dataset of the paper), seeded with Li+ and EC:

```bash
python examples/run_li_ec_demo.py --gmc /path/to/RNMC/build/GMC
```

Expect roughly 5-15 minutes on a laptop. Each iteration prints the
candidates tested, reactions kept, atom-mapping rejections, and species
promoted; the loop stops on its own when an iteration promotes no new
species and ends with a summary of the generated network. Outputs land
in `examples/out_li_ec/`: the network (`rn.sqlite`), `final_summary.json`
with the full config and per-iteration statistics, and the per-iteration
folders. For product and pathway reports on a converged network, use
`tools/production_kmc.py` as described below. The promotion threshold
epsilon defaults to `1e-2` here; pass `--eps` to explore other values
(smaller epsilon grows a larger network and takes longer).

## Quickstart (your own chemistry)

```bash
forge --config my_run.json
```

`forge` is the console entry point for `python -m forge.run_seeded`; the two
invocations are equivalent. The config is a flat JSON file mapping
one-to-one onto `forge.config.SeededConfig`; every field is documented
inline in `forge/config.py`, and keys starting with `_` are treated as
comments and stripped before parsing. At minimum you must fill in
`mol_entries_pickle`, `out_dir`, `gmc_binary_path`, `electron_free_energy`,
and a seed-species specification. Seeds can be given
as direct pickle indices, as `{formula, charge, count}` queries (recommended,
resolved to the lowest-G coordimer at startup), or as XYZ files. If the
species pickle does not exist yet, set `species_json_path` and
`solvation_environment` and FORGE runs `chemistry_lib.species_filter` at
startup to build it.

## Typical workflow

**1. Build the network.**

```bash
forge --config my_run.json
```

Each iteration writes an `iter_NN/` folder under `out_dir` (buckets, initial
state, GMC trajectories, per-iteration stats). The cumulative network
accumulates in `out_dir/rn.sqlite`; `core_state.json` holds the promoted-core
state and makes killed runs resumable by re-invoking the same command;
`final_summary.json` records the full config plus per-iteration statistics.

**2. Production kMC.** The per-iteration runs are screening kMC with modest
statistics (`n_simulations`, default 2000). For final analysis, run one
high-statistics production kMC on the converged network:

```bash
python -m tools.production_kmc --config my_run.json \
    --n-simulations 50000 --step-cutoff 200000 --subdir final_kmc
```

This rebuilds the initial state from the same config, runs GMC against the
final `rn.sqlite`, and regenerates the sink, tally, species, and per-sink
pathway reports under `out_dir/final_kmc/`.


## Attribution

The inline atom-to-atom mapper in `aam_src/` follows the Ising-model
formulation of Ali et al. and is bundled here in adapted, self-contained
form. The species and reaction decision trees, filtering, and kMC analysis in
`chemistry_lib/` follow the HiPRGen framework of Barter et al.
(https://github.com/BlauGroup/HiPRGen). The kinetic Monte Carlo engine is GMC
from RNMC. (https://github.com/BlauGroup/RNMC).
