"""Unit tests for the core pipeline invariants.

Run with: myenv/bin/python -m pytest   (pythonpath=src is set in pyproject.toml)
"""
import numpy as np
import pandas as pd
import pytest

import config
from functions.duplicate_entries import duplicate_entries
from functions.preprocessing import load_and_preprocess
from feature_engineering.generate_stats import generate_stats
from feature_engineering.calculate_elo import calculate_elo, INITIAL_ELO
from feature_engineering.head2head import add_h2h_stats
from model_evaluations import chronological_pair_split, time_series_pair_cv


# --------------------------------------------------------------------------- #
# duplicate_entries: mirroring must swap ids/elo and negate diffs
# --------------------------------------------------------------------------- #
def test_duplicate_entries_mirror():
    df = pd.DataFrame({
        "player_id": [1, 3],
        "opponent_id": [2, 4],
        "rank_diff": [10.0, -5.0],
        "surface_elo_diff": [30.0, -8.0],
        "player_elo_before": [1200.0, 900.0],
        "opponent_elo_before": [1000.0, 1100.0],
        "win_loss": [1, 1],
    })
    out = duplicate_entries(df)

    assert len(out) == 2 * len(df)
    # original at even rows, mirror at odd rows (adjacent pairs)
    orig, mirror = out.iloc[0], out.iloc[1]
    assert mirror["win_loss"] == 1 - orig["win_loss"] == 0
    assert mirror["rank_diff"] == -orig["rank_diff"]
    assert mirror["surface_elo_diff"] == -orig["surface_elo_diff"]
    # ids and elo are swapped, not negated
    assert mirror["player_id"] == orig["opponent_id"]
    assert mirror["opponent_id"] == orig["player_id"]
    assert mirror["player_elo_before"] == orig["opponent_elo_before"]
    assert mirror["opponent_elo_before"] == orig["player_elo_before"]


def test_duplicate_entries_pair_invariant():
    df = pd.DataFrame({
        "player_id": range(5), "opponent_id": range(5, 10),
        "rank_diff": np.arange(5.0), "win_loss": [1] * 5,
    })
    y = duplicate_entries(df)["win_loss"].values
    # every adjacent pair sums to exactly 1
    assert np.array_equal(y[0::2] + y[1::2], np.ones(len(df)))


# --------------------------------------------------------------------------- #
# generate_stats: rolling averages must use only prior matches (no leakage)
# --------------------------------------------------------------------------- #
def test_generate_stats_no_leakage():
    # A plays three matches in order; B and C appear once each.
    df = pd.DataFrame({
        "player_id":   [1, 1, 1],   # A always the (winner) player
        "opponent_id": [2, 3, 2],   # B, C, B
        "w_ace": [10, 20, 30], "l_ace": [2, 4, 6],
        "w_df":  [1, 2, 3],    "l_df":  [0, 1, 2],
        "w_bpSaved": [5, 6, 7], "l_bpSaved": [1, 1, 1],
    })
    out = generate_stats(df, window=10)

    # First match: A has no history -> NaN -> ace_diff NaN
    assert np.isnan(out.loc[0, "player_ace_avg"])
    assert np.isnan(out.loc[0, "ace_diff"])
    # Second match: A's only prior ace count is 10
    assert out.loc[1, "player_ace_avg"] == 10
    # Third match: A's prior ace counts are 10 and 20 -> mean 15
    assert out.loc[2, "player_ace_avg"] == 15
    # B (opponent in match 0 and 2): in match 2, B's prior is the l_ace from match 0 = 2
    assert out.loc[2, "opponent_ace_avg"] == 2


# --------------------------------------------------------------------------- #
# calculate_elo: no magic number, ratings start at INITIAL_ELO, winner gains
# --------------------------------------------------------------------------- #
def _elo_df(n):
    return pd.DataFrame({
        "player_id":   [1, 2] * (n // 2) + [1] * (n % 2),
        "opponent_id": [2, 1] * (n // 2) + [2] * (n % 2),
        "surface": [0] * n,
        "win_loss": [1] * n,
    })


@pytest.mark.parametrize("n", [1, 3, 6, 17])
def test_calculate_elo_runs_for_any_size(tmp_path, monkeypatch, n):
    # Old code hard-coded 27672 rows; this must work for arbitrary n.
    monkeypatch.setitem(config.PATHS, "elo_ratings", tmp_path / "elo.csv")
    out = calculate_elo(_elo_df(n))
    assert len(out) == n
    assert {"player_elo_before", "opponent_elo_before", "surface_elo_diff"} <= set(out.columns)


def test_calculate_elo_semantics(tmp_path, monkeypatch):
    monkeypatch.setitem(config.PATHS, "elo_ratings", tmp_path / "elo.csv")
    out = calculate_elo(_elo_df(2))
    # First match: both start at INITIAL_ELO, surface diff 0
    assert out.loc[0, "player_elo_before"] == INITIAL_ELO
    assert out.loc[0, "opponent_elo_before"] == INITIAL_ELO
    assert out.loc[0, "surface_elo_diff"] == 0.0
    # Second match: player_id 1 won match 0, so its rating is now above 1000
    # (player of match 1 is id 2, the previous loser -> below 1000)
    assert out.loc[1, "opponent_elo_before"] > INITIAL_ELO   # id 1 as opponent
    assert out.loc[1, "player_elo_before"] < INITIAL_ELO     # id 2 as player


# --------------------------------------------------------------------------- #
# load_and_preprocess: works for both tours (winner/loser rename + level map)
# --------------------------------------------------------------------------- #
def _write_raw(tmp_path, cols: dict):
    path = tmp_path / "raw.csv"
    pd.DataFrame(cols).to_csv(path, index=False)
    return path


def test_preprocess_renames_winner_loser_and_keeps_serve_stats(tmp_path):
    # Sackmann-style raw file (as freshly downloaded for WTA): winner_*/loser_*.
    path = _write_raw(tmp_path, {
        "tourney_level": ["PM"], "round": ["F"], "best_of": [3], "surface": ["Hard"],
        "winner_rank": [3], "loser_rank": [10],
        "winner_rank_points": [4000], "loser_rank_points": [2000],
        "winner_age": [25.0], "loser_age": [28.0],
        "w_ace": [5], "l_ace": [3],   # serve stats must survive the rename
    })
    out = load_and_preprocess(path)

    # winner_/loser_ renamed to player_/opponent_ ...
    assert {"player_rank", "opponent_rank", "player_age", "opponent_age"} <= set(out.columns)
    assert not any(c.startswith(("winner_", "loser_")) for c in out.columns)
    # ... but w_*/l_* serve-stat columns are left untouched.
    assert "w_ace" in out.columns and "l_ace" in out.columns
    # WTA tourney_level code 'PM' maps to a non-zero match_importance (5*7*3).
    assert out.loc[0, "match_importance"] == 5 * 7 * 3
    assert out.loc[0, "win_loss"] == 1


def test_preprocess_idempotent_on_player_opponent_columns(tmp_path):
    # Already-renamed file (the existing ATP set) must pass through untouched.
    path = _write_raw(tmp_path, {
        "tourney_level": ["G"], "round": ["QF"], "best_of": [5], "surface": ["Clay"],
        "player_rank": [1], "opponent_rank": [50],
        "player_rank_points": [9000], "opponent_rank_points": [1200],
        "player_age": [22.0], "opponent_age": [30.0],
    })
    out = load_and_preprocess(path)
    assert out.loc[0, "match_importance"] == 6 * 5 * 5  # G=6, QF=5, best_of=5
    assert out.loc[0, "rank_diff"] == 49               # -(1 - 50)


def test_calculate_elo_writes_to_explicit_path(tmp_path):
    out_path = tmp_path / "nested" / "wta_elo.csv"
    calculate_elo(_elo_df(2), elo_output_path=out_path)
    assert out_path.exists()


def test_head2head_pre_match():
    df = pd.DataFrame({"player_id": [1, 1], "opponent_id": [2, 2]})
    out = add_h2h_stats(df)
    # First meeting: 0; second meeting after 1 win: diff should be 1
    assert out.loc[0, "h2h_diff"] == 0
    assert out.loc[1, "h2h_diff"] == 1


# --------------------------------------------------------------------------- #
# Evaluation splits: no pair split, train strictly before test
# --------------------------------------------------------------------------- #
def test_chronological_pair_split_even_boundary():
    tr, te = chronological_pair_split(10, test_frac=0.3)
    assert len(tr) % 2 == 0 and len(te) % 2 == 0
    assert tr[-1] < te[0]                      # chronological, no overlap
    assert list(tr) + list(te) == list(range(10))


def test_time_series_pair_cv_no_leakage():
    n_rows = 2 * 100
    for tr, te in time_series_pair_cv(n_rows, n_splits=5):
        assert set(tr).isdisjoint(te)          # no row in both sides
        assert tr.max() < te.min()             # train strictly before test (time)
        # pairs stay intact: both 2p and 2p+1 present together on each side
        for idx in (tr, te):
            pairs = idx // 2
            assert len(idx) == 2 * len(set(pairs))
