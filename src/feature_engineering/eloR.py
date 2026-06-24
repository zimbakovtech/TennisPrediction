import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
from pathlib import Path

KEY_FEATURES = [
    'surface', 'player_rank', 'opponent_rank',
    'player_rank_points', 'opponent_rank_points'
]
CURRENT_DIR = Path.cwd()

dataset_with_elo = []

# -------------------------
# Read files and prepare data
# -------------------------
files = sorted([f"{CURRENT_DIR}/data/raw/{f}" for f in os.listdir(f'{CURRENT_DIR}/data/raw') if re.match(r'atp_matches_*', f)])
if not files:
    print("No files matching pattern 'atp_matches_*' found in current directory.")

df_list = [pd.read_csv(f) for f in files]
matches_raw = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
matches_raw = matches_raw.dropna(subset=KEY_FEATURES)
matches_raw = matches_raw[matches_raw['surface'] != 'Carpet']

matches_raw['tourney_date'] = pd.to_datetime(matches_raw['tourney_date'].astype(str), format='%Y%m%d', errors='coerce').dt.date
matches_raw['match_num'] = pd.to_numeric(matches_raw['match_num'], errors='coerce')
matches_raw = matches_raw.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)

# -------------------------
# Globals: player Elo storage and match counts
# -------------------------
firstDate = date(1900, 1, 1)
playersToElo = {}
matchesCount = {}

SURFACE_TYPES = ('grass', 'clay', 'hard')
SURFACE_ALIASES = {
    'grass': 'grass',
    'clay': 'clay',
    'hard': 'hard',
    'hard indoor': 'hard',
    'indoor hard': 'hard'
}
playersSurfaceToElo = {surface: {} for surface in SURFACE_TYPES}
matchesSurfaceCount = {surface: {} for surface in SURFACE_TYPES}

TREND_LOOKBACK_DAYS = 90

# -------------------------
# Helper functions (Elo logic)
# -------------------------
def normalize_surface(surface_value):
    """Return canonical lowercase surface key (grass/clay/hard) or None if unsupported."""
    if not isinstance(surface_value, str):
        return None
    normalized = surface_value.strip().lower()
    return SURFACE_ALIASES.get(normalized)


def update_matches_count(playerA, playerB, counts=None):
    """Increment matches count dictionary for both players (initialize to 0 if missing)."""
    counts = matchesCount if counts is None else counts
    if playerA not in counts or counts[playerA] is None:
        counts[playerA] = 0
    if playerB not in counts or counts[playerB] is None:
        counts[playerB] = 0
    counts[playerA] += 1
    counts[playerB] += 1

def get_latest_ranking(store, player):
    """Return the most recent Elo rating for the player within the provided store."""
    history = store.get(player)
    if not history:
        return 1500.0
    return history[-1]['ranking']


def get_ranking_before_date(history, cutoff_date, default=1500.0):
    """Return the player's Elo at or before the provided cutoff date."""
    if not history:
        return default

    for entry in reversed(history):
        entry_date = entry.get('date')
        if entry_date and entry_date <= cutoff_date:
            return entry['ranking']

    return default


def update_elo(plToElo, row, match_counts=None):
    """
    Update Elo for two players based on a match result.
    Mirrors the R updateElo function exactly:
      - initialize to 1500 if missing (with firstDate, num=0)
      - compute expected scores eA, eB
      - sA/sB from winner
      - kA = 250/((matchesCount[playerA]+5)^0.4)
      - k multiplier = 1.1 if level == "G" else 1
      - r_new = r + (k*kA) * (s - e)
      - append new row to each player's history
    """

    match_counts = matchesCount if match_counts is None else match_counts

    playerA = row['player_name']
    playerB = row['opponent_name']
    level = row['tourney_level']
    matchDate = row['tourney_date']
    matchNum = row['match_num']

    if playerA not in plToElo or not plToElo[playerA]:
        plToElo[playerA] = [{'ranking': 1500.0, 'date': firstDate, 'num': 0}]
    if playerB not in plToElo or not plToElo[playerB]:
        plToElo[playerB] = [{'ranking': 1500.0, 'date': firstDate, 'num': 0}]

    rA = plToElo[playerA][-1]['ranking']
    rB = plToElo[playerB][-1]['ranking']

    eA = 1.0 / (1.0 + 10 ** ((rB - rA) / 400.0))
    eB = 1.0 / (1.0 + 10 ** ((rA - rB) / 400.0))

    sA, sB = 1.0, 0.0

    kA = 250.0 / ((match_counts.get(playerA, 0) + 5.0) ** 0.4)
    kB = 250.0 / ((match_counts.get(playerB, 0) + 5.0) ** 0.4)
    k = 1.1 if (level == "G") else 1.0

    rA_new = rA + (k * kA) * (sA - eA)
    rB_new = rB + (k * kB) * (sB - eB)

    plToElo[playerA].append({'ranking': float(rA_new), 'date': matchDate, 'num': matchNum})
    plToElo[playerB].append({'ranking': float(rB_new), 'date': matchDate, 'num': matchNum})
    
    return rA, rB

def compute_elo_by_row(row):
    player_elo_before, opponent_elo_before = update_elo(playersToElo, row)
    return player_elo_before, opponent_elo_before

def update_matches_count_by_row(row, counts=None):
    update_matches_count(row['player_name'], row['opponent_name'], counts)
    return 0


def compute_surface_elo_by_row(row, surface_key):
    return update_elo(
        playersSurfaceToElo[surface_key],
        row,
        match_counts=matchesSurfaceCount[surface_key]
    )

# -------------------------
# Top-level computeElo (same flow: precompute counts, then compute Elo)
# -------------------------
def computeElo():
    for _, row in matches_raw.iterrows():
        surface_key = normalize_surface(row['surface'])

        player_surface_pre = {
            surface: get_latest_ranking(playersSurfaceToElo[surface], row['player_name'])
            for surface in SURFACE_TYPES
        }
        opponent_surface_pre = {
            surface: get_latest_ranking(playersSurfaceToElo[surface], row['opponent_name'])
            for surface in SURFACE_TYPES
        }

        trend_cutoff = row['tourney_date'] - timedelta(days=TREND_LOOKBACK_DAYS)
        player_baseline_elo = get_ranking_before_date(playersToElo.get(row['player_name']), trend_cutoff)
        opponent_baseline_elo = get_ranking_before_date(playersToElo.get(row['opponent_name']), trend_cutoff)

        update_matches_count_by_row(row)
        if surface_key in SURFACE_TYPES:
            update_matches_count_by_row(row, matchesSurfaceCount[surface_key])

        player_elo_before, opponent_elo_before = compute_elo_by_row(row)

        if surface_key in SURFACE_TYPES:
            compute_surface_elo_by_row(row, surface_key)

        player_elo_trend = round(player_elo_before - player_baseline_elo, 5)
        opponent_elo_trend = round(opponent_elo_before - opponent_baseline_elo, 5)

        dataset_with_elo.append({
            'tourney_date': row['tourney_date'],
            'surface': row['surface'],
            'player_id': row['player_id'],
            'opponent_id': row['opponent_id'],
            'player_name': row['player_name'],
            'opponent_name': row['opponent_name'],
            'player_elo_before': player_elo_before,
            'opponent_elo_before': opponent_elo_before,
            'player_elo_trend': player_elo_trend,
            'opponent_elo_trend': opponent_elo_trend,
            'player_elo_before_grass': (player_surface_pre['grass'] + player_elo_before) / 2.0,
            'player_elo_before_clay': (player_surface_pre['clay'] + player_elo_before) / 2.0,
            'player_elo_before_hard': (player_surface_pre['hard'] + player_elo_before) / 2.0,
            'opponent_elo_before_grass': (opponent_surface_pre['grass'] + opponent_elo_before) / 2.0,
            'opponent_elo_before_clay': (opponent_surface_pre['clay'] + opponent_elo_before) / 2.0,
            'opponent_elo_before_hard': (opponent_surface_pre['hard'] + opponent_elo_before) / 2.0
        })
        
    return playersToElo

# -------------------------
# If this script is invoked directly, run computeElo() and print a short summary
# -------------------------
if __name__ == "__main__":
    computeElo()
    elo_data = []
    for player, history in playersToElo.items():
        for entry in history:
            elo_data.append({
                'player_name': player,
                'ranking': entry['ranking'],
                'date': entry['date'],
                'num': entry['num']
            })
    historic_elo_df = pd.DataFrame(elo_data)
    historic_elo_df.to_csv('players_elo_history.csv', index=False)

    elo_df = pd.DataFrame(dataset_with_elo)
    elo_df.to_csv('matches_with_elo.csv', index=False)
    