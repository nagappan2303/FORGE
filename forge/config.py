"""Configuration dataclass for a seeded-iterative FORGE run."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_nproc() -> int:
    try:
        return int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    except ValueError:
        return 1


@dataclass
class SeededConfig:
    # --- required inputs ----------------------------------------------------
    mol_entries_pickle: str
    out_dir: str
    electron_free_energy: float             # solvent/cation specific

    # --- seed-species specification (pick ONE mode) -------------------------
    #
    # Mode 1 (direct): supply integer indices already known.
    seed_species: List[int] = field(default_factory=list)
    initial_state_counts: Dict[int, int] = field(default_factory=dict)
    # species_id -> molecule count at t=0.  Keys must match seed_species.
    
    # Mode 2 (by formula+charge): FORGE resolves the lowest-G coordimer for
    # each entry against the pickle after species_filter runs. Populates
    # seed_species + initial_state_counts internally. Each entry must be
    # {"formula": "Li1", "charge": 1, "count": 600}.
    seed_species_query: Optional[List[Dict[str, Any]]] = None
    
    # Mode 3 (by XYZ file): resolve indices against the pickle by matching
    # geometry + charge + spin. Each record is
    # {"xyz": "/abs/path.xyz", "charge": 0, "spin": 1, "count": 9000}.
    seed_species_xyz: Optional[List[Dict[str, Any]]] = None

    # --- species filter (optional: run species_filter at startup) -----------
    species_json_path: Optional[str] = None
    # Path to raw species JSON. If set and `mol_entries_pickle` does not yet
    # exist, FORGE runs species_filter at startup to produce the pickle.
    # Leave None to use a pre-built pickle.

    solvation_environment: Optional[str] = None
    # Required when species_json_path is set: 'li_ec' or 'Na_ec' (see
    # chemistry_lib/constants.py). Selects the solvation-corrected
    # free-energy reference used by species_filter.

    # --- thermodynamics ------------------------------------------------------
    temperature: float = 298.15

    # --- iteration controls --------------------------------------------------
    max_iterations: int = 12
    promotion_flux_metric: str = "produced"     # "produced" | "touched" | "consumed"

    # Promotion criterion: scale-invariant fraction of total flux.
    # This is the normalized flux threshold epsilon of the FORGE paper's
    # promotion rule. The flux metric (default 'produced') is an event count
    # summed over every trajectory, so a raw count scales with n_simulations,
    # seed counts, and step_cutoff, and is non-comparable across compositions. 
    # We instead promote species whose
    # share of total flux clears a fraction:
    #     promote if flux[s][metric] / sum_t flux[t][metric] >= promotion_flux_fraction
    # This removes n_simulations / seed-count / step_cutoff dependence and keeps the
    # core lean across scales.
    promotion_flux_fraction: float = 1.0e-3

    convergence_new_species_threshold: int = 0  # stop when <= this many promoted

    # --- kMC seeding strategy ------------------------------------------------
    exploratory_initial_count: int = 0
    # If > 0, seed every species that appears in this iter's RN at this
    # count. Creates noise in flux counts; keep at 0 for pure mass-action
    # physics with realistic seed counts instead.
    exploratory_only_core_reactants: bool = True

    # --- kMC (short runs per iter) ------------------------------------------
    n_simulations: int = 100
    step_cutoff: int = 200000
    kmc_seed: int = 1000
    gmc_binary_path: str = "./GMC"
    gmc_thread_count: int = 32

    # --- reaction-decision-tree selector ------------------------------------
    reaction_tree: str = "default"
    # Only "default" is supported (chemistry_lib.reaction_questions.
    # default_reaction_decision_tree).

    # --- seeded reaction generator (composition-indexed product lookup) -----
    enforce_reactant_in_core: bool = True
    # Keep True for seeded discipline: skip bucket entries where NEITHER
    # species touches core (belt-and-suspenders with the bucketer).
    commit_freq: int = 5000
    progress_every: int = 50000

    # Enumeration backend for the per-iteration reaction generator:
    #   "fork" (default): single-node multiprocessing.Pool. Memory scales with
    #          nproc × composition-index size and with the bucket count, so it
    #          OOMs once a core grows large (buckets ≈ core × species_set).
    #   "mpi" : dispatcher-worker across nodes (forge.mpi_reaction_generator,
    #          launched via mpi_launcher). Bounded per-rank memory; control
    #          per-node footprint via ranks-per-node. AAM stays inline per rank
    #          via a persistent posix_spawn'd helper (no fork-from-MPI). Use for
    #          large multi-component chemistries (~10k species).
    enumeration_backend: str = "fork"            # "fork" | "mpi"
    # Launcher used when enumeration_backend == "mpi". {ranks}/{cmd} are
    # substituted; e.g. "srun -n {ranks} {cmd}" or "mpirun -n {ranks} {cmd}".
    # SLURM jobs typically prefer srun (Slurm PMI) for rank placement.
    mpi_launcher: str = "srun -n {ranks} {cmd}"
    # Total MPI ranks (1 dispatcher + N-1 workers). 0 => read $SLURM_NTASKS.
    mpi_ranks: int = 0

    # Where the composition->product-pairs index lives (only used by the "mpi"
    # backend):
    #   "memory" (default): each worker builds the O(N^2) pair_by_comp index in
    #          RAM (~3.5 GB/worker for a large chemistry -> forces multi-node).
    #   "disk" : the index is built ONCE to <out_dir>/composition_index.sqlite and
    #          every worker queries it (read-only), holding only mol_entries +
    #          a connection (~tens of MB beyond mol_entries). Lets a large chemistry run
    #          single-node with many ranks; also skips the per-iter rebuild.
    enumeration_index: str = "memory"            # "memory" | "disk"

    nproc: int = field(default_factory=_default_nproc)
    # Defaults to SLURM_CPUS_PER_TASK when set, else 1. The fork-based
    # multiprocessing.Pool inherits mol_entries + tree via copy-on-write
    # memory, so there is no serialization overhead per call.
    pool_chunksize: int = 1
    # With imap_unordered chunksize=N, each worker holds N tasks at a time;
    # if one task hits a slow AAM call, the other N-1 are stalled. The
    # default 1 maximizes responsiveness; raise only if your AAM bill is
    # cheap and IPC dominates.

    # --- Atom-mapping (AAM) filter ------------------------------------------
    # How AAM is applied each iteration (only matters when aam_filter=True):
    #   "inline" (default): AAM runs inside the enumeration worker, per reaction,
    #            right after the structural decision tree.
    #   "batch" : DECOUPLED per-iteration flow: (A) structural enumeration into
    #            a per-iter candidates DB, (B) one batch AAM pass over the unique
    #            candidates (de-duplicated, fork-Pool, on-disk cache), (C) append
    #            the AAM-allowed rows to the cumulative rn.sqlite. AAM still runs
    #            BEFORE kMC each iteration (gating preserved), but enumeration
    #            stays pure-structural so it can use enumeration_backend="mpi"
    #            without an inline mapper in the hot loop. Same final rn.sqlite.
    aam_mode: str = "inline"                     # "inline" | "batch"

    aam_filter: bool = True
    # If True, every reaction surviving the decision tree is also AAM-
    # validated before being appended to rn.sqlite. Guarantees clean kMC
    # dynamics at the cost of an AAM call per reaction.

    aam_timeout_sec: int = 120
    # Per-reaction mapper hard timeout (SIGKILL).

    aam_keep_on_timeout: bool = True
    # Fail-open on indecisive mapper outcomes (timeout / no-mapping / error):
    # the reaction is KEPT so pathological mapper cases don't silently delete
    # chemistry.

    aam_cache_path: Optional[str] = None
    # Disk-backed AAM result cache (sqlite, WAL). If None, defaults to
    # <out_dir>/aam_cache.sqlite (set by the iteration driver). Keyed on
    # canonical reactant.smi >> product.smi; survives across iters and
    # across runs that share an out_dir, so canonically-identical reactions
    # are mapped at most once.

    aam_pre_filter: bool = True
    # Run the cheap covalent-bond-count delta check before invoking the
    # mapper. Rejects reactions whose per-bond-type count differences sum above prefilter_bond_delta_cutoff (typed; elementary bound 2)
    # as non-elementary. O(1) per reaction.

    prefilter_bond_delta_cutoff: int = 3
    # Structural pre-AAM filter: drops reactions where the absolute change
    # in total covalent-edge count between reactants and products exceeds
    # this threshold. The mapper itself enforces the elementary-step
    # criterion (bond edits n_b + n_f <= 2 with shared atom) internally;
    # this cutoff is an inexpensive early-out.

    # --- Iteration safety ----------------------------------------------------
    iter_wall_timeout_sec: int = 0
    # If a single iter's reaction generation exceeds this wall-clock budget,
    # finalize with whatever AAM-clean reactions have been kept so far and
    # proceed to kMC + promotion. 0 disables the budget.

    iter_progress_log_sec: int = 300
    # AAM-rate watchdog log every N seconds during reaction generation.

    # --- Auto-report generation (post production kMC) -----------------------
    pathway_ev_cutoff: float = 4.0
    # Generate a per-sink pathway report for every sink species whose
    # expected_value >= this cutoff.

    n_pathways_per_sink: int = 100
    # Forwarded to mc_analysis.generate_pathway_report as the
    # number_of_pathways argument.

    end_of_run_reports: bool = True
    # Generate the sink/tally/species/pathway reports automatically when the
    # loop finishes. tools/production_kmc.py emits the same reports from a
    # high-statistics rerun regardless of this setting.

    seed: int = 42

    def as_dict(self) -> Dict[str, Any]:
        return {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in self.__dict__.items()
            if not callable(v)
        }

    def out_path(self) -> Path:
        p = Path(self.out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
