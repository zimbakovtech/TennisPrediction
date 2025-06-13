import pandas as pd
import numpy as np
from functions import utils


def calculate_elo(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    y = utils.process_y()

    # Step 2: Initialize ELO ratings for all players
    all_players = pd.concat([new_df['player_id'], new_df['opponent_id']]).unique()
    elo_ratings = {player: 1000.0 for player in all_players}
    match_counts = {player: 0 for player in all_players}

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
        # Extract player and winner IDs
        player = row['player_id']
        opponent = row['opponent_id']
        winner = row['player_id'] if label == 1 else row['opponent_id']
        
        # Get current ELO ratings before the match
        elo1 = elo_ratings[player]
        elo2 = elo_ratings[opponent]
        
        # Store these ELO ratings in the lists
        player_elo_before.append(elo1)
        opponent_elo_before.append(elo2)

        # Calculate expected scores
        # E1 is the expected score for player, based on the ELO difference
        E1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
        E2 = 1 - E1  # E2 is the complement, as it's a zero-sum game
        
        # Determine actual scores based on the winner
        if winner == player:
            S1 = 1  # Player 1 wins
            S2 = 0
        elif winner == opponent:
            S1 = 0  # Player 2 wins
            S2 = 1
        else:
            raise ValueError("Winner ID does not match either player ID")
        
        # Update ELO ratings using the formula: new_elo = old_elo + K * (actual - expected)
        new_elo1 = elo1 + K_max * (S1 - E1)
        new_elo2 = elo2 + K_max * (S2 - E2)

        match_counts[player] += 1
        match_counts[opponent] += 1
        
        if match_counts[player] < decay_threshold:
            new_elo1 = (match_counts[player] * new_elo1 + M * initial_elo) / (match_counts[player] + M)
        if match_counts[opponent] < decay_threshold:
            new_elo2 = (match_counts[opponent] * new_elo2 + M * initial_elo) / (match_counts[opponent] + M)
        
        # Update the dictionary with the new ELO ratings
        if index % 2 == 1:
            elo_ratings[player] = new_elo1
            elo_ratings[opponent] = new_elo2

    # Step 6: Add the ELO ratings to the DataFrame
    new_df['player_elo_before'] = player_elo_before
    new_df['opponent_elo_before'] = opponent_elo_before

    # Step 7: Create a DataFrame for final ELO ratings
    elo_df = pd.DataFrame(list(elo_ratings.items()), columns=['player_id', 'elo'])

    # Sort by ELO in descending order
    elo_df = elo_df.sort_values(by='elo', ascending=False).reset_index(drop=True)

    # Step 8: Save the sorted ELO ratings to a separate CSV file
    elo_df.to_csv('data/players/player_elo_ratings.csv', index=False)

    return new_df
