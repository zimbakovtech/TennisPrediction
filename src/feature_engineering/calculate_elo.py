import pandas as pd
import numpy as np
from functions import utils

def calculate_elo(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    y = utils.process_y()

    # Step 2: Initialize ELO ratings for all players on all surfaces
    all_players = pd.concat([new_df['player_id'], new_df['opponent_id']]).unique()
    surfaces = [1, 2, 3]  # 0: hard, 1: clay, 2: grass
    elo_ratings = {surface: {player: 1000.0 for player in all_players} for surface in surfaces}
    match_counts = {surface: {player: 0 for player in all_players} for surface in surfaces}

    # Step 3: Prepare lists to store ELO ratings before each match
    player_elo_before = []
    opponent_elo_before = []

    # Step 4: Set the K-factor
    K_max = 32
    decay_threshold = 5
    M = 5
    initial_elo = 1000

    # Step 5: Process each match in chronological order
    for (index, row), label in zip(new_df.iterrows(), y):
        surface = row['surface']
        player = row['player_id']
        opponent = row['opponent_id']
        winner = row['player_id'] if label == 1 else row['opponent_id']

        # Get current ELO ratings before the match for the surface
        elo1 = elo_ratings[surface][player]
        elo2 = elo_ratings[surface][opponent]

        # Store these ELO ratings in the lists
        player_elo_before.append(elo1)
        opponent_elo_before.append(elo2)

        # Calculate expected scores
        E1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
        E2 = 1 - E1

        # Determine actual scores based on the winner
        if winner == player:
            S1 = 1
            S2 = 0
        elif winner == opponent:
            S1 = 0
            S2 = 1
        else:
            raise ValueError("Winner ID does not match either player ID")

        # Update ELO ratings using the formula: new_elo = old_elo + K * (actual - expected)
        new_elo1 = elo1 + K_max * (S1 - E1)
        new_elo2 = elo2 + K_max * (S2 - E2)

        match_counts[surface][player] += 1
        match_counts[surface][opponent] += 1

        if match_counts[surface][player] < decay_threshold:
            new_elo1 = (match_counts[surface][player] * new_elo1 + M * initial_elo) / (match_counts[surface][player] + M)
        if match_counts[surface][opponent] < decay_threshold:
            new_elo2 = (match_counts[surface][opponent] * new_elo2 + M * initial_elo) / (match_counts[surface][opponent] + M)

        # Update the dictionary with the new ELO ratings for the surface
        if index % 2 == 1:
            elo_ratings[surface][player] = round(new_elo1, 5)
            elo_ratings[surface][opponent] = round(new_elo2, 5)

    # Step 6: Add the ELO ratings to the DataFrame
    new_df['player_elo'] = player_elo_before
    new_df['opponent_elo'] = opponent_elo_before
    # new_df['elo_diff'] = new_df['player_elo'] - new_df['opponent_elo']

    # Step 7: Create a DataFrame for final ELO ratings (for each surface)
    elo_rows = []
    for surface in surfaces:
        for player, elo in elo_ratings[surface].items():
            elo_rows.append({'player_id': player, 'surface': surface, 'elo': elo})
    elo_df = pd.DataFrame(elo_rows)

    # Sort by ELO in descending order
    elo_df = elo_df.sort_values(by=['surface', 'elo'], ascending=[True, False]).reset_index(drop=True)

    # Step 8: Save the sorted ELO ratings to a separate CSV file
    elo_df.to_csv('data/players/player_elo_ratings.csv', index=False)

    return new_df
