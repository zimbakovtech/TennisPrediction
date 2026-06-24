#!/usr/bin/env python3
"""Convert `winner_id`/`loser_id` in `atp_matches_2025.csv` to 6-digit IDs.

`data/raw/atp_matches_2025.csv` uses 4-character alphanumeric IDs (e.g. CD85).
`matches_with_elo.csv` contains 6-digit numeric IDs paired with player names.

This script rewrites ONLY the `winner_id` and `loser_id` fields by matching on
`winner_name` and `loser_name` using a name->id mapping derived from
`matches_with_elo.csv`.

It is careful to:
- keep the same columns and row order
- preserve all other fields exactly as strings
- only replace when the name maps to a unique 6-digit ID

By default, writes an output file. With `--inplace`, it replaces the input file
atomically and writes a `.bak` backup next to it.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple


_SIX_DIGIT_RE = re.compile(r"^\d{6}$")


def _norm_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return ""

    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s\-']+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _is_six_digit(value: str) -> bool:
    return bool(_SIX_DIGIT_RE.match((value or "").strip()))


@dataclass
class Report:
    replaced_winner_id: int = 0
    replaced_loser_id: int = 0
    already_six_digit_winner_id: int = 0
    already_six_digit_loser_id: int = 0
    missing_winner_name: int = 0
    missing_loser_name: int = 0
    unknown_winner_name: int = 0
    unknown_loser_name: int = 0
    ambiguous_winner_name: int = 0
    ambiguous_loser_name: int = 0


def _load_name_to_ids_from_elo(elo_csv: str) -> Dict[str, Set[str]]:
    mapping: Dict[str, Set[str]] = defaultdict(set)

    with open(elo_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"player_id", "player_name", "opponent_id", "opponent_name"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{elo_csv} must contain columns {sorted(required)}; got {reader.fieldnames}"
            )

        for row in reader:
            pid = (row.get("player_id") or "").strip()
            pname = (row.get("player_name") or "").strip()
            if pname and _is_six_digit(pid):
                mapping[_norm_name(pname)].add(pid)

            oid = (row.get("opponent_id") or "").strip()
            oname = (row.get("opponent_name") or "").strip()
            if oname and _is_six_digit(oid):
                mapping[_norm_name(oname)].add(oid)

    return mapping


def _unique_id(name_to_ids: Dict[str, Set[str]], name: str) -> Optional[str]:
    ids = name_to_ids.get(_norm_name(name), set())
    if len(ids) == 1:
        return next(iter(ids))
    return None


def convert(
    *,
    input_csv: str,
    output_csv: str,
    elo_csv: str,
) -> Report:
    report = Report()

    name_to_ids = _load_name_to_ids_from_elo(elo_csv)

    with open(input_csv, "r", newline="") as fin:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            raise ValueError(f"{input_csv} appears to have no header")

        required = {"winner_id", "winner_name", "loser_id", "loser_name"}
        if not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{input_csv} must contain columns {sorted(required)}; got {reader.fieldnames}"
            )

        with open(output_csv, "w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, lineterminator="\n")
            writer.writeheader()

            for row in reader:
                wid = (row.get("winner_id") or "").strip()
                if _is_six_digit(wid):
                    report.already_six_digit_winner_id += 1
                else:
                    wname = (row.get("winner_name") or "").strip()
                    if not wname:
                        report.missing_winner_name += 1
                    else:
                        ids = name_to_ids.get(_norm_name(wname), set())
                        if not ids:
                            report.unknown_winner_name += 1
                        elif len(ids) > 1:
                            report.ambiguous_winner_name += 1
                        else:
                            new_wid = next(iter(ids))
                            if wid != new_wid:
                                row["winner_id"] = new_wid
                                report.replaced_winner_id += 1

                lid = (row.get("loser_id") or "").strip()
                if _is_six_digit(lid):
                    report.already_six_digit_loser_id += 1
                else:
                    lname = (row.get("loser_name") or "").strip()
                    if not lname:
                        report.missing_loser_name += 1
                    else:
                        ids = name_to_ids.get(_norm_name(lname), set())
                        if not ids:
                            report.unknown_loser_name += 1
                        elif len(ids) > 1:
                            report.ambiguous_loser_name += 1
                        else:
                            new_lid = next(iter(ids))
                            if lid != new_lid:
                                row["loser_id"] = new_lid
                                report.replaced_loser_id += 1

                writer.writerow(row)

    return report


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=os.path.join("data", "raw", "atp_matches_2025.csv"),
        help="Input ATP matches 2025 CSV (default: data/raw/atp_matches_2025.csv)",
    )
    parser.add_argument(
        "--elo",
        default="matches_with_elo.csv",
        help="Source CSV containing 6-digit IDs + names (default: matches_with_elo.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: <input>.converted.csv)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Replace input file atomically and write a .bak backup",
    )
    args = parser.parse_args(argv)

    input_path = args.input
    elo_path = args.elo
    output_path = args.output or (input_path + ".converted.csv")

    if args.inplace:
        # Write to temp in same directory, then replace.
        base_dir = os.path.dirname(os.path.abspath(input_path))
        tmp_path = os.path.join(base_dir, os.path.basename(input_path) + ".tmp")
        backup_path = input_path + ".bak"

        report = convert(input_csv=input_path, output_csv=tmp_path, elo_csv=elo_path)
        if os.path.exists(backup_path):
            raise SystemExit(f"Refusing to overwrite existing backup: {backup_path}")
        os.replace(input_path, backup_path)
        os.replace(tmp_path, input_path)
        print(f"Wrote backup: {backup_path}")
    else:
        if os.path.abspath(input_path) == os.path.abspath(output_path):
            raise SystemExit("Refusing to overwrite input without --inplace")
        report = convert(input_csv=input_path, output_csv=output_path, elo_csv=elo_path)

    print("Conversion complete")
    print("- replaced winner_id:", report.replaced_winner_id)
    print("- replaced loser_id:", report.replaced_loser_id)
    print("- already 6-digit winner_id:", report.already_six_digit_winner_id)
    print("- already 6-digit loser_id:", report.already_six_digit_loser_id)
    print("- unknown winner_name:", report.unknown_winner_name)
    print("- unknown loser_name:", report.unknown_loser_name)
    print("- ambiguous winner_name:", report.ambiguous_winner_name)
    print("- ambiguous loser_name:", report.ambiguous_loser_name)

    if report.unknown_winner_name or report.unknown_loser_name or report.ambiguous_winner_name or report.ambiguous_loser_name:
        print(
            "WARNING: Some names did not map uniquely; their IDs were left unchanged.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
