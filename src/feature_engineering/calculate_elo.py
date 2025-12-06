import pandas as pd
import numpy as np
from functions import utils

def calculate_elo(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    y = np.array([1, 0] * 101884).tolist()

    # Step 2: Initialize ELO ratings for all players (overall and per surface)
    all_players = pd.concat([new_df['player_id'], new_df['opponent_id']]).unique()
    elo_ratings = {player: 1000.0 for player in all_players}
    match_counts = {player: 0 for player in all_players}
    # Surface ELOs: 0=hard, 1=clay, 2=grass
    surface_elo_ratings = {surface: {player: 1000.0 for player in all_players} for surface in [0, 1, 2]}
    surface_match_counts = {surface: {player: 0 for player in all_players} for surface in [0, 1, 2]}

    # Step 3: Prepare lists to store ELO ratings before each match
    player_elo_before = []
    opponent_elo_before = []
    player_surface_elo_before = []
    opponent_surface_elo_before = []
    surface_elo_diff = []
    elo_diff = []

    # Step 4: Set the K-factor
    K_max = 32
    decay_threshold = 5
    M = 5
    initial_elo = 1000

    # Step 5: Process each match in chronological order
    for (index, row), label in zip(new_df.iterrows(), y):
        player = row['player_id']
        opponent = row['opponent_id']
        winner = row['player_id'] if label == 1 else row['opponent_id']
        surface = row['surface']

        # Get current ELO ratings before the match
        elo1 = elo_ratings[player]
        elo2 = elo_ratings[opponent]
        surf_elo1 = surface_elo_ratings[surface][player]
        surf_elo2 = surface_elo_ratings[surface][opponent]

        # Store these ELO ratings in the lists
        player_elo_before.append(elo1)
        opponent_elo_before.append(elo2)
        player_surface_elo_before.append(surf_elo1)
        opponent_surface_elo_before.append(surf_elo2)
        elo_diff.append(round(elo1 - elo2, 5))
        surface_elo_diff.append(round(surf_elo1 - surf_elo2, 5))

        # Calculate expected scores (overall)
        E1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
        E2 = 1 - E1

        # Calculate expected scores (surface)
        surf_E1 = 1 / (1 + 10 ** ((surf_elo2 - surf_elo1) / 400))
        surf_E2 = 1 - surf_E1

        # Determine actual scores
        if winner == player:
            S1 = 1
            S2 = 0
        elif winner == opponent:
            S1 = 0
            S2 = 1
        else:
            raise ValueError("Winner ID does not match either player ID")

        # Update ELO ratings (overall)
        new_elo1 = elo1 + K_max * (S1 - E1)
        new_elo2 = elo2 + K_max * (S2 - E2)
        match_counts[player] += 1
        match_counts[opponent] += 1
        if match_counts[player] < decay_threshold:
            new_elo1 = (match_counts[player] * new_elo1 + M * initial_elo) / (match_counts[player] + M)
        if match_counts[opponent] < decay_threshold:
            new_elo2 = (match_counts[opponent] * new_elo2 + M * initial_elo) / (match_counts[opponent] + M)

        # Update ELO ratings (surface)
        new_surf_elo1 = surf_elo1 + K_max * (S1 - surf_E1)
        new_surf_elo2 = surf_elo2 + K_max * (S2 - surf_E2)
        surface_match_counts[surface][player] += 1
        surface_match_counts[surface][opponent] += 1
        if surface_match_counts[surface][player] < decay_threshold:
            new_surf_elo1 = (surface_match_counts[surface][player] * new_surf_elo1 + M * initial_elo) / (surface_match_counts[surface][player] + M)
        if surface_match_counts[surface][opponent] < decay_threshold:
            new_surf_elo2 = (surface_match_counts[surface][opponent] * new_surf_elo2 + M * initial_elo) / (surface_match_counts[surface][opponent] + M)

        # Update the dictionaries with the new ELO ratings
        if index % 2 == 1:
            elo_ratings[player] = round(new_elo1, 5)
            elo_ratings[opponent] = round(new_elo2, 5)
            surface_elo_ratings[surface][player] = round(new_surf_elo1, 5)
            surface_elo_ratings[surface][opponent] = round(new_surf_elo2, 5)

    # Step 6: Add the ELO ratings to the DataFrame
    new_df['player_elo_before'] = player_elo_before
    new_df['opponent_elo_before'] = opponent_elo_before
    # new_df['elo_diff'] = elo_diff
    # new_df['player_surface_elo_before'] = player_surface_elo_before
    # new_df['opponent_surface_elo_before'] = opponent_surface_elo_before
    new_df['surface_elo_diff'] = surface_elo_diff

    # Step 7: Create a DataFrame for final ELO ratings
    elo_df = pd.DataFrame(list(elo_ratings.items()), columns=['player_id', 'elo'])
    elo_df = elo_df.sort_values(by='elo', ascending=False).reset_index(drop=True)
    elo_df.to_csv('data/players/player_elo_ratings.csv', index=False)

    return new_df
