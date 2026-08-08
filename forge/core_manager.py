"""Tracks the growing core-species set across iterations."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set


@dataclass
class CoreState:
    """Mutable state carried across iterations."""

    core_ids: Set[int] = field(default_factory=set)
    per_iter_promoted: List[List[int]] = field(default_factory=list)
    per_iter_flux_summary: List[dict] = field(default_factory=list)
    per_iter_bucket_stats: List[dict] = field(default_factory=list)
    per_iter_rxn_counts: List[int] = field(default_factory=list)
    iteration: int = 0

    # mapping from promoted species_id -> iteration where it was first promoted
    promoted_at: Dict[int, int] = field(default_factory=dict)

    def promote(self, species_ids, iteration: int):
        ids = [int(s) for s in species_ids if int(s) not in self.core_ids]
        for s in ids:
            self.core_ids.add(s)
            self.promoted_at[s] = iteration
        self.per_iter_promoted.append(ids)
        return ids

    def snapshot(self) -> dict:
        return {
            "iteration": self.iteration,
            "core_size": len(self.core_ids),
            "core_ids_sorted": sorted(int(i) for i in self.core_ids),
            "per_iter_promoted": self.per_iter_promoted,
            "per_iter_flux_summary": self.per_iter_flux_summary,
            "per_iter_bucket_stats": self.per_iter_bucket_stats,
            "per_iter_rxn_counts": self.per_iter_rxn_counts,
            "promoted_at": {int(k): int(v) for k, v in self.promoted_at.items()},
        }

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.snapshot(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CoreState":
        d = json.loads(Path(path).read_text())
        cs = cls(
            core_ids=set(d["core_ids_sorted"]),
            per_iter_promoted=d["per_iter_promoted"],
            per_iter_flux_summary=d["per_iter_flux_summary"],
            per_iter_bucket_stats=d["per_iter_bucket_stats"],
            per_iter_rxn_counts=d["per_iter_rxn_counts"],
            iteration=d["iteration"],
            promoted_at={int(k): int(v) for k, v in d["promoted_at"].items()},
        )
        return cs
