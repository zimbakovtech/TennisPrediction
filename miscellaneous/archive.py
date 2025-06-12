# def generate_stats(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
#     # Columns for which to calculate rolling averages
#     stats_cols = [
#         "w_ace",
#         "l_ace",
#         "w_df",
#         "l_df",
#         "w_bpSaved",
#         "l_bpSaved",
#     ]

#     # Group by player, shift to exclude current match, then compute rolling mean
#     rolling_avgs = (
#         df
#         .groupby("player_id")[stats_cols]
#         .apply(
#             lambda grp: grp
#             .shift(1)  # exclude the current match from its own average
#             .rolling(window=window, min_periods=1)
#             .mean()
#             .round(2)
#         )
#         .reset_index(level=0, drop=True)
#     )

#     # Rename columns to indicate they're averages
#     rolling_avgs.columns = [f"{col}_avg" for col in rolling_avgs.columns]

#     # Concatenate rolling averages alongside the original data
#     result_df = pd.concat([df, rolling_avgs], axis=1)

#     return result_df



# df['w_1stIn_avg'] = w_1stIn_avgs
    # df['l_1stIn_avg'] = l_1stIn_avgs
    # df['w_1stWon_avg'] = w_1stWon_avgs
    # df['l_1stWon_avg'] = l_1stWon_avgs
    # df['w_2ndWon_avg'] = w_2ndWon_avgs
    # df['l_2ndWon_avg'] = l_2ndWon_avgs
    # df['w_bpFaced_avg'] = w_bpFaced_avgs
    # df['l_bpFaced_avg'] = l_bpFaced_avgs
    # df['w_svpt_avg'] = w_svpt_avgs
    # df['l_svpt_avg'] = l_svpt_avgs

    # Calculate percentage metrics
    # df['w_bpSavedPer'] = (df['w_bpSaved_avg'] / df['w_bpFaced_avg']).replace([np.inf, -np.inf], np.nan).round(2)
    # df['l_bpSavedPer'] = (df['l_bpSaved_avg'] / df['l_bpFaced_avg']).replace([np.inf, -np.inf], np.nan).round(2)
    # df['w_1stPer'] = (df['w_1stWon_avg'] / df['w_1stIn_avg']).replace([np.inf, -np.inf], np.nan).round(2)
    # df['l_1stPer'] = (df['l_1stWon_avg'] / df['l_1stIn_avg']).replace([np.inf, -np.inf], np.nan).round(2)
    # df['w_2ndPer'] = (df['w_2ndWon_avg'] / (df['w_svpt_avg'] - df['w_1stIn_avg'])).replace([np.inf, -np.inf], np.nan).round(2)
    # df['l_2ndPer'] = (df['l_2ndWon_avg'] / (df['l_svpt_avg'] - df['l_1stIn_avg'])).replace([np.inf, -np.inf], np.nan).round(2)

    # l_1stIn_avgs.append(round(opponent_1stIn.mean(), 2) if not opponent_1stIn.empty else None)
        # l_1stWon_avgs.append(round(opponent_1stWon.mean(), 2) if not opponent_1stWon.empty else None)
        # l_2ndWon_avgs.append(round(opponent_2ndWon.mean(), 2) if not opponent_2ndWon.empty else None)
        # l_bpFaced_avgs.append(round(opponent_bpFaced.mean(), 2) if not opponent_bpFaced.empty else None)
        # l_svpt_avgs.append(round(opponent_svpt.mean(), 2) if not opponent_svpt.empty else None)

        # New metrics for opponent
        # opponent_1stIn = opponent_matches.apply(
        #     lambda r: r['w_1stIn'] if r['player_id'] == opponent_id else r['l_1stIn'], axis=1
        # ).tail(window)
        # opponent_1stWon = opponent_matches.apply(
        #     lambda r: r['w_1stWon'] if r['player_id'] == opponent_id else r['l_1stWon'], axis=1
        # ).tail(window)
        # opponent_2ndWon = opponent_matches.apply(
        #     lambda r: r['w_2ndWon'] if r['player_id'] == opponent_id else r['l_2ndWon'], axis=1
        # ).tail(window)
        # opponent_bpFaced = opponent_matches.apply(
        #     lambda r: r['w_bpFaced'] if r['player_id'] == opponent_id else r['l_bpFaced'], axis=1
        # ).tail(window)
        # opponent_svpt = opponent_matches.apply(
        #     lambda r: r['w_svpt'] if r['player_id'] == opponent_id else r['l_svpt'], axis=1
        # ).tail(window)

        # w_1stIn_avgs.append(round(player_1stIn.mean(), 2) if not player_1stIn.empty else None)
        # w_1stWon_avgs.append(round(player_1stWon.mean(), 2) if not player_1stWon.empty else None)
        # w_2ndWon_avgs.append(round(player_2ndWon.mean(), 2) if not player_2ndWon.empty else None)
        # w_bpFaced_avgs.append(round(player_bpFaced.mean(), 2) if not player_bpFaced.empty else None)
        # w_svpt_avgs.append(round(player_svpt.mean(), 2) if not player_svpt.empty else None)

    # New metrics for player
        # player_1stIn = player_matches.apply(
        #     lambda r: r['w_1stIn'] if r['player_id'] == player_id else r['l_1stIn'], axis=1
        # ).tail(window)
        # player_1stWon = player_matches.apply(
        #     lambda r: r['w_1stWon'] if r['player_id'] == player_id else r['l_1stWon'], axis=1
        # ).tail(window)
        # player_2ndWon = player_matches.apply(
        #     lambda r: r['w_2ndWon'] if r['player_id'] == player_id else r['l_2ndWon'], axis=1
        # ).tail(window)
        # player_bpFaced = player_matches.apply(
        #     lambda r: r['w_bpFaced'] if r['player_id'] == player_id else r['l_bpFaced'], axis=1
        # ).tail(window)
        # player_svpt = player_matches.apply(
        #     lambda r: r['w_svpt'] if r['player_id'] == player_id else r['l_svpt'], axis=1
        # ).tail(window)

        # New metrics
    # w_1stIn_avgs, l_1stIn_avgs = [], []
    # w_1stWon_avgs, l_1stWon_avgs = [], []
    # w_2ndWon_avgs, l_2ndWon_avgs = [], []
    # w_bpFaced_avgs, l_bpFaced_avgs = [], []
    # w_svpt_avgs, l_svpt_avgs = [], []


# Fill NaN values in the specified columns with 0
    # columns_to_fill = [
    #     # 'w_1stIn_avg', 'l_1stIn_avg', 'w_1stWon_avg', 'l_1stWon_avg',
    #     # 'w_2ndWon_avg', 'l_2ndWon_avg', 'w_bpFaced_avg', 'l_bpFaced_avg',
    #     # 'w_svpt_avg', 'l_svpt_avg', 
    #     # 'w_bpSavedPer', 'l_bpSavedPer',
    #     # 'w_1stPer', 'l_1stPer', 'w_2ndPer', 'l_2ndPer'
    # ]
    # for col in columns_to_fill:
    #     averaged_df[col] = averaged_df.get(col, 0).fillna(0)