import pandas as pd
from collections import defaultdict, deque

def add_h2h_stats(df: pd.DataFrame, streak_length: int = 5) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)

    # h2h[player][opponent] = (player's wins vs opponent) - (opponent's wins vs player)
    h2h = defaultdict(lambda: defaultdict(int))
    h2h_diffs = []

    # track recent results per player: +1 for win, -1 for loss, capped at streak_length
    recent_results = defaultdict(lambda: deque(maxlen=streak_length))
    player_streaks = []
    opponent_streaks = []

    for _, row in df.iterrows():
        player = row['player_id']      # winner
        opponent = row['opponent_id']  # loser

        # compute current h2h diff
        diff = h2h[player][opponent]
        h2h_diffs.append(diff)

        # compute player streak: look at recent_results[player]
        p_hist = recent_results[player]
        if p_hist:
            # determine sign of latest result
            last_sign = p_hist[-1]
            # count consecutive entries from end matching last_sign
            count = 0
            for result in reversed(p_hist):
                if result == last_sign:
                    count += 1
                else:
                    break
            player_streaks.append(last_sign * count)
        else:
            player_streaks.append(0)

        # compute opponent streak: look at recent_results[opponent]
        o_hist = recent_results[opponent]
        if o_hist:
            last_sign = o_hist[-1]
            count = 0
            for result in reversed(o_hist):
                if result == last_sign:
                    count += 1
                else:
                    break
            opponent_streaks.append(last_sign * count)
        else:
            opponent_streaks.append(0)

        # update h2h map after this match
        h2h[player][opponent] += 1
        h2h[opponent][player] -= 1

        # record this match result in recent history
        recent_results[player].append(1)    # win
        recent_results[opponent].append(-1) # loss

    # assign new columns
    df['h2h_diff'] = h2h_diffs
    df['player_streak'] = player_streaks
    df['opponent_streak'] = opponent_streaks
    df['streak_diff'] = df['player_streak'] - df['opponent_streak']
    return df