#!/usr/bin/env python3
"""Merge fix3 generated H5 paths by scenario quota and scenario+seed uniqueness."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_QUOTAS = {
    "hole_late_move_stop": 70,
    "hole_late_constant": 90,
    "hole_late_reverse": 100,
    "hole_late_sine": 90,
    "hole_late_continuous_insert": 120,
    "hole_late_fast_shift": 120,
    "none": 160,
    "peg_drop": 150,
    "peg_disturb": 100,
}


@dataclass(frozen=True)
class Candidate:
    scenario: str
    seed: int
    policy_seed: int | None
    path: Path

    @property
    def key(self) -> tuple[str, int, int | None]:
        return self.scenario, self.seed, self.policy_seed


FILENAME_PATTERN = re.compile(r"^(?P<scenario>.+)_seed(?P<seed>\d+)(?:_pseed(?P<policy_seed>\d+))?_idx\d+\.h5$")


def parse_quotas(text: str) -> dict[str, int]:
    if not text:
        return dict(DEFAULT_QUOTAS)
    quotas: dict[str, int] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"quota entry must be SCENARIO=COUNT, got {item!r}")
        key, value = item.split("=", 1)
        quotas[key.strip()] = int(value)
    unknown = sorted(set(quotas) - set(DEFAULT_QUOTAS))
    if unknown:
        raise ValueError(f"unknown scenarios in quotas: {unknown}")
    missing = sorted(set(DEFAULT_QUOTAS) - set(quotas))
    if missing:
        raise ValueError(f"missing scenarios in quotas: {missing}")
    return quotas


def candidate_from_path(path: Path) -> Candidate | None:
    match = FILENAME_PATTERN.match(path.name)
    if not match:
        return None
    scenario = match.group("scenario")
    if scenario not in DEFAULT_QUOTAS:
        return None
    policy_seed_text = match.group("policy_seed")
    return Candidate(
        scenario=scenario,
        seed=int(match.group("seed")),
        policy_seed=None if policy_seed_text is None else int(policy_seed_text),
        path=path.resolve(),
    )


def scan_root(root: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    for path in sorted(root.rglob("*.h5")):
        candidate = candidate_from_path(path)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def scan_paths_file(paths_file: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    base = paths_file.resolve().parent
    for raw in paths_file.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        path = Path(text)
        if not path.is_absolute():
            path = (base / path).resolve() if not path.exists() else path.resolve()
        candidate = candidate_from_path(path)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-root", action="append", required=True, type=Path)
    parser.add_argument("--source-paths-file", action="append", default=[], type=Path)
    parser.add_argument("--quotas", default="")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    quotas = parse_quotas(args.quotas)
    selected: dict[str, list[Candidate]] = {scenario: [] for scenario in quotas}
    seen: set[tuple[str, int, int | None]] = set()
    raw_counts = {scenario: 0 for scenario in quotas}
    duplicate_count = 0
    source_roots = [root.resolve() for root in args.source_root]
    source_paths_files = [path.resolve() for path in args.source_paths_file]

    for source_name, candidates in [
        *[(str(root), scan_root(root)) for root in source_roots],
        *[(str(paths_file), scan_paths_file(paths_file)) for paths_file in source_paths_files],
    ]:
        for candidate in candidates:
            raw_counts[candidate.scenario] += 1
            if candidate.key in seen:
                duplicate_count += 1
                continue
            seen.add(candidate.key)
            if len(selected[candidate.scenario]) < quotas[candidate.scenario]:
                selected[candidate.scenario].append(candidate)

    selected_counts = {scenario: len(items) for scenario, items in selected.items()}
    missing = {scenario: quotas[scenario] - count for scenario, count in selected_counts.items()}
    missing = {scenario: count for scenario, count in missing.items() if count > 0}
    total_selected = sum(selected_counts.values())
    expected_total = sum(quotas.values())
    if missing and not args.allow_incomplete:
        raise SystemExit(
            "insufficient unique H5 candidates: "
            + json.dumps({"missing": missing, "selected_counts": selected_counts}, sort_keys=True)
        )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected_paths = [candidate.path for scenario in quotas for candidate in selected[scenario]]
    (output_root / "fix3_h5_paths.txt").write_text(
        "".join(f"{path}\n" for path in selected_paths),
        encoding="utf-8",
    )
    manifest = {
        "schema": "fix3_full1000_unique_merge_v1",
        "output_root": str(output_root),
        "source_roots": [str(root) for root in source_roots],
        "source_paths_files": [str(path) for path in source_paths_files],
        "quotas": quotas,
        "raw_counts": raw_counts,
        "selected_counts": selected_counts,
        "total_selected": total_selected,
        "expected_total": expected_total,
        "duplicate_scenario_seed_count": duplicate_count,
        "missing": missing,
        "stop_for_user_approval": total_selected == expected_total and not missing,
        "notes": [
            "Selection is by scenario+seed uniqueness, or scenario+seed+policy_seed when a policy seed is present; raw H5 count is not evidence of unique data.",
            "When policy_rng_seed is present in the filename, uniqueness is scenario+seed+policy_seed.",
            "This file-level merge does not replace H5 structural/physical audits.",
            "Do not start WAM export, render expansion, controller integration, or Cosmos3 SFT before user approval.",
        ],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
