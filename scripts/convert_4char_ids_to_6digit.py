#!/usr/bin/env python3
"""Convert non-6-digit player IDs to 6-digit ATP IDs.

This repo contains match-style CSVs that use `player_id` / `opponent_id`.
If one CSV uses non-6-digit IDs (e.g. legacy 4-char codes) while another
uses the 6-digit numeric ATP IDs, this script rewrites only the ID columns
based on player names.

Design goals:
- Do NOT edit input files by default. Always write a new output file.
- Preserve all other columns/values exactly as strings (no pandas).
- Only modify `player_id` / `opponent_id` values when they are not already
  6-digit numeric strings.

Typical usage:
  python scripts/convert_4char_ids_to_6digit.py \
    --input path/to/legacy.csv \
    --output path/to/legacy_converted.csv \
    --atp-players miscellaneous/atp_players.csv

Optionally, you may also pass a matches file (like data/raw/atp_matches_2024.csv)
to add extra name->id evidence (helpful if `atp_players.csv` has name variants).

  python scripts/convert_4char_ids_to_6digit.py \
    --input path/to/legacy.csv \
    --output path/to/legacy_converted.csv \
    --atp-players miscellaneous/atp_players.csv \
    --ref-matches data/raw/atp_matches_2024.csv
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
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


_SIX_DIGIT_RE = re.compile(r"^\d{6}$")


def _norm_name(name: str) -> str:
    """Normalize a human name for joining.

    - Unicode normalize + strip accents
    - Lowercase
    - Collapse whitespace
    - Remove most punctuation
    """

    name = (name or "").strip()
    if not name:
        return ""

    # Strip accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))

    # Normalize punctuation/spaces
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s\-']+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _is_six_digit(value: str) -> bool:
    return bool(_SIX_DIGIT_RE.match((value or "").strip()))


@dataclass
class MappingReport:
    replaced_player_id: int = 0
    replaced_opponent_id: int = 0
    already_six_digit_player_id: int = 0
    already_six_digit_opponent_id: int = 0
    missing_player_name: int = 0
    missing_opponent_name: int = 0
    unknown_player_name: int = 0
    unknown_opponent_name: int = 0
    ambiguous_player_name: int = 0
    ambiguous_opponent_name: int = 0


def _load_atp_players_name_to_ids(atp_players_csv: str) -> Dict[str, Set[str]]:
    """Build normalized full_name -> {player_id} mapping from atp_players.csv."""

    mapping: Dict[str, Set[str]] = defaultdict(set)
    with open(atp_players_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"player_id", "name_first", "name_last"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{atp_players_csv} must contain columns {sorted(required)}; got {reader.fieldnames}"
            )

        for row in reader:
            pid = (row.get("player_id") or "").strip()
            if not _is_six_digit(pid) and not pid.isdigit():
                # Defensive: ignore weird IDs.
                continue

            first = (row.get("name_first") or "").strip()
            last = (row.get("name_last") or "").strip()
            full = f"{first} {last}".strip()
            key = _norm_name(full)
            if key:
                mapping[key].add(pid)

    return mapping


def _load_ref_matches_name_to_ids(ref_matches_csv: str) -> Dict[str, Set[str]]:
    """Build normalized player_name/opponent_name -> {id} mapping from a matches CSV."""

    mapping: Dict[str, Set[str]] = defaultdict(set)
    with open(ref_matches_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return mapping

        # Accept headers with possible whitespace.
        header = {h.strip(): h for h in reader.fieldnames}
        needed = {"player_id", "player_name", "opponent_id", "opponent_name"}
        if not needed.issubset(set(header.keys())):
            return mapping

        pid_col = header["player_id"]
        pname_col = header["player_name"]
        oid_col = header["opponent_id"]
        oname_col = header["opponent_name"]

        for row in reader:
            pid = (row.get(pid_col) or "").strip()
            pname = (row.get(pname_col) or "").strip()
            if pid and pname and pid.isdigit():
                mapping[_norm_name(pname)].add(pid)

            oid = (row.get(oid_col) or "").strip()
            oname = (row.get(oname_col) or "").strip()
            if oid and oname and oid.isdigit():
                mapping[_norm_name(oname)].add(oid)

    return mapping


def _merge_mappings(*maps: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    merged: Dict[str, Set[str]] = defaultdict(set)
    for m in maps:
        for name_key, ids in m.items():
            merged[name_key].update(ids)
    return merged


def _pick_unique_id(name_to_ids: Dict[str, Set[str]], name: str) -> Optional[str]:
    ids = name_to_ids.get(_norm_name(name) or "", set())
    if len(ids) == 1:
        return next(iter(ids))
    return None


def convert_csv_ids(
    *,
    input_csv: str,
    output_csv: str,
    name_to_ids: Dict[str, Set[str]],
    player_id_col: str = "player_id",
    opponent_id_col: str = "opponent_id",
    player_name_col: str = "player_name",
    opponent_name_col: str = "opponent_name",
) -> MappingReport:
    """Convert IDs for a CSV while preserving all other fields."""

    report = MappingReport()

    with open(input_csv, "r", newline="") as fin:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            raise ValueError(f"{input_csv} appears to have no header")

        # Tolerate whitespace in headers by mapping requested->actual.
        header = {h.strip(): h for h in reader.fieldnames}
        for required in (player_id_col, opponent_id_col, player_name_col, opponent_name_col):
            if required not in header:
                raise ValueError(
                    f"{input_csv} must contain '{required}' column (whitespace-insensitive). Found: {reader.fieldnames}"
                )

        pid_h = header[player_id_col]
        oid_h = header[opponent_id_col]
        pname_h = header[player_name_col]
        oname_h = header[opponent_name_col]

        with open(output_csv, "w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                # Player
                pid = (row.get(pid_h) or "").strip()
                if _is_six_digit(pid):
                    report.already_six_digit_player_id += 1
                else:
                    pname = (row.get(pname_h) or "").strip()
                    if not pname:
                        report.missing_player_name += 1
                    else:
                        ids = name_to_ids.get(_norm_name(pname) or "", set())
                        if not ids:
                            report.unknown_player_name += 1
                        elif len(ids) > 1:
                            report.ambiguous_player_name += 1
                        else:
                            new_pid = next(iter(ids))
                            if pid != new_pid:
                                row[pid_h] = new_pid
                                report.replaced_player_id += 1

                # Opponent
                oid = (row.get(oid_h) or "").strip()
                if _is_six_digit(oid):
                    report.already_six_digit_opponent_id += 1
                else:
                    oname = (row.get(oname_h) or "").strip()
                    if not oname:
                        report.missing_opponent_name += 1
                    else:
                        ids = name_to_ids.get(_norm_name(oname) or "", set())
                        if not ids:
                            report.unknown_opponent_name += 1
                        elif len(ids) > 1:
                            report.ambiguous_opponent_name += 1
                        else:
                            new_oid = next(iter(ids))
                            if oid != new_oid:
                                row[oid_h] = new_oid
                                report.replaced_opponent_id += 1

                writer.writerow(row)

    return report


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input CSV with legacy IDs")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--atp-players",
        default=os.path.join("miscellaneous", "atp_players.csv"),
        help="Path to atp_players.csv (default: miscellaneous/atp_players.csv)",
    )
    parser.add_argument(
        "--ref-matches",
        default=None,
        help="Optional matches CSV with 6-digit ids + names (e.g. data/raw/atp_matches_2024.csv)",
    )
    args = parser.parse_args(argv)

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        raise SystemExit("Refusing to overwrite input. Choose a different --output.")

    if not os.path.exists(args.input):
        raise SystemExit(f"Input not found: {args.input}")
    if not os.path.exists(args.atp_players):
        raise SystemExit(f"atp_players not found: {args.atp_players}")

    base = _load_atp_players_name_to_ids(args.atp_players)
    ref = _load_ref_matches_name_to_ids(args.ref_matches) if args.ref_matches else {}
    name_to_ids = _merge_mappings(base, ref)

    report = convert_csv_ids(
        input_csv=args.input,
        output_csv=args.output,
        name_to_ids=name_to_ids,
    )

    print("Conversion complete")
    print("- replaced player_id:", report.replaced_player_id)
    print("- replaced opponent_id:", report.replaced_opponent_id)
    print("- already 6-digit player_id:", report.already_six_digit_player_id)
    print("- already 6-digit opponent_id:", report.already_six_digit_opponent_id)
    print("- missing player_name:", report.missing_player_name)
    print("- missing opponent_name:", report.missing_opponent_name)
    print("- unknown player_name:", report.unknown_player_name)
    print("- unknown opponent_name:", report.unknown_opponent_name)
    print("- ambiguous player_name:", report.ambiguous_player_name)
    print("- ambiguous opponent_name:", report.ambiguous_opponent_name)

    if (
        report.unknown_player_name
        or report.unknown_opponent_name
        or report.ambiguous_player_name
        or report.ambiguous_opponent_name
    ):
        print(
            "WARNING: Some IDs could not be mapped uniquely. Those rows were left unchanged.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
