import pandas as pd
import numpy as np
from functions import utils

def half_y():
    return np.array([1] * 15980)


def calculate_elo(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    y = half_y()

    all_players = pd.concat([new_df['player_id'], new_df['opponent_id']]).unique()
    elo_ratings = {player: 1000.0 for player in all_players}  # Start all players at 1000.0
    match_counts = {player: 0 for player in all_players}  # Track matches played

    # Step 3: Prepare lists to store ELO ratings before each match
    player_elo_before_list = []
    opponent_elo_before_list = []

    # Step 4: Define K-factor parameters and decay settings
    K_max = 32  # Maximum K-factor
    scale = 10  # Controls how quickly K decreases with matches played
    decay_threshold = 5  # Apply decay for players with fewer than this many matches
    M = 5  # Weight of initial rating for decay
    initial_elo = 1000.0  # Initial ELO for regression

    # Step 5: Process each match in the sorted DataFrame
    for index, row in new_df.iterrows():
        # Extract player and opponent IDs
        player = row['player_id']
        opponent = row['opponent_id']
        
        # Get current ELO ratings before the match
        elo1 = elo_ratings[player]
        elo2 = elo_ratings[opponent]
        
        # Store these ELO ratings in the lists
        player_elo_before_list.append(elo1)
        opponent_elo_before_list.append(elo2)
        
        # Get current match counts
        matches1 = match_counts[player]
        matches2 = match_counts[opponent]
        
        # Calculate variable K-factors based on matches played
        K1 = K_max / (1 + matches1 / scale)
        K2 = K_max / (1 + matches2 / scale)
        
        # Calculate expected scores
        E1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
        E2 = 1 - E1
        
        # Set actual scores: player_id always wins, opponent_id always loses
        S1 = 1  # Player 1 wins
        S2 = 0  # Player 2 loses
        
        # Update ELO ratings
        new_elo1 = elo1 + K1 * (S1 - E1)
        new_elo2 = elo2 + K2 * (S2 - E2)
        
        # Apply regression to the mean for players with few matches
        match_counts[player] += 1
        match_counts[opponent] += 1
        
        if match_counts[player] < decay_threshold:
            new_elo1 = (match_counts[player] * new_elo1 + M * initial_elo) / (match_counts[player] + M)
        if match_counts[opponent] < decay_threshold:
            new_elo2 = (match_counts[opponent] * new_elo2 + M * initial_elo) / (match_counts[opponent] + M)
        
        # Update the dictionary with new ELO ratings
        elo_ratings[player] = new_elo1
        elo_ratings[opponent] = new_elo2

    # Step 6: Add ELO ratings to the DataFrame
    new_df['player_elo_before'] = player_elo_before_list
    new_df['opponent_elo_before'] = opponent_elo_before_list

    # utils.create_elo_csv(elo_ratings)

    return new_df
