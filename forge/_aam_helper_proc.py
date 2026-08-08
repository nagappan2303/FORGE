"""Persistent AAM helper subprocess for forge.mpi_reaction_generator.

A plain (NON-MPI) process spawned once per worker rank via posix_spawn. It loads
mol_entries once and then runs the REAL aam_check_reaction for each reaction read
from stdin, emitting "1" (KEEP) / "0" (DROP) on stdout.

Because this process never calls MPI_Init, aam_check_reaction's internal
os.fork()+SIGKILL hard-timeout is safe here, so AAM behaviour is byte-identical
to single-node FORGE. The parent MPI worker provides a second-line watchdog
(SIGKILL+respawn) only for the rare case the helper itself wedges.

argv: <mol_pickle> <cache_path|''> <timeout_s> <keep_on_timeout 0|1>
      <pre_filter 0|1> <max_bond_delta>

Protocol:
  - emits one "READY\n" line after mol_entries is loaded (parent waits for it).
  - then, per stdin line (JSON {"reactants":[inds], "products":[inds]}),
    emits "1\n" or "0\n".
"""
import json
import pickle
import sys


def main():
    mol_pickle = sys.argv[1]
    cache_path = sys.argv[2] or None
    timeout_s = int(sys.argv[3])
    keep_on_timeout = sys.argv[4] == "1"
    pre_filter = sys.argv[5] == "1"
    max_bond_delta = int(sys.argv[6])

    with open(mol_pickle, "rb") as f:
        mol_entries = pickle.load(f)

    from forge.aam_filter import aam_check_reaction

    sys.stdout.write("READY\n"); sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            reaction = {"reactants": req["reactants"], "products": req["products"]}
            allowed, _info = aam_check_reaction(
                reaction, mol_entries,
                timeout_s=timeout_s, keep_on_timeout=keep_on_timeout,
                cache_path=cache_path, pre_filter=pre_filter,
                max_bond_delta=max_bond_delta,
            )
        except Exception:
            allowed = keep_on_timeout            # fail-open on any error
        sys.stdout.write(("1" if allowed else "0") + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
