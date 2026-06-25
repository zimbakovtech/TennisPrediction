import pandas as pd

from config import PATHS

# Elo configuration
K_FACTOR = 32
INITIAL_ELO = 1500.0
DECAY_THRESHOLD = 5      # matches below which a player's rating is pulled to INITIAL_ELO
DECAY_WEIGHT = 5         # strength of that pull (M in the blended-average formula)
SURFACES = (0, 1, 2)     # 0=hard, 1=clay, 2=grass


def _expected(elo_a: float, elo_b: float) -> float:
    """Standard logistic Elo expected score for player A."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))


def _decay(new_elo: float, count: int) -> float:
    """Blend a young player's rating toward INITIAL_ELO until DECAY_THRESHOLD."""
    if count < DECAY_THRESHOLD:
        return (count * new_elo + DECAY_WEIGHT * INITIAL_ELO) / (count + DECAY_WEIGHT)
    return new_elo


def calculate_elo(df: pd.DataFrame, elo_output_path=None) -> pd.DataFrame:
    """Compute pre-match (global and per-surface) Elo ratings.

    Runs on the *pre-mirror* frame, where each row is one real match and the
    ``player_*`` columns are the winner (``win_loss == 1``). Ratings are read
    *before* the update and stored, so they never leak the current result.

    The input MUST be in chronological order. The mirrored perspective is
    produced later by ``duplicate_entries`` (which swaps the two elo columns
    and negates ``surface_elo_diff``), so this function does not need to know
    about mirroring -- no magic row counts, no index parity tricks.

    The final global ratings are written to ``elo_output_path`` (defaults to
    ``PATHS['elo_ratings']``); pass a per-tour path to keep ATP and WTA dumps
    from overwriting each other.
    """
    df = df.copy()

    players = df["player_id"].to_numpy()
    opponents = df["opponent_id"].to_numpy()
    surfaces = df["surface"].to_numpy()
    wins = df["win_loss"].to_numpy()

    all_players = pd.concat([df["player_id"], df["opponent_id"]]).unique()
    elo = {p: INITIAL_ELO for p in all_players}
    counts = {p: 0 for p in all_players}
    surf_elo = {s: {p: INITIAL_ELO for p in all_players} for s in SURFACES}
    surf_counts = {s: {p: 0 for p in all_players} for s in SURFACES}

    player_elo_before = []
    opponent_elo_before = []
    surface_elo_diff = []

    for i in range(len(df)):
        p, o, s, win = players[i], opponents[i], surfaces[i], wins[i]

        elo_p, elo_o = elo[p], elo[o]
        player_elo_before.append(elo_p)
        opponent_elo_before.append(elo_o)

        # S1 is 1 when the player (winner pre-mirror) wins.
        s1 = 1.0 if win == 1 else 0.0
        s2 = 1.0 - s1

        # --- Global Elo update ---
        e_p = _expected(elo_p, elo_o)
        counts[p] += 1
        counts[o] += 1
        elo[p] = round(_decay(elo_p + K_FACTOR * (s1 - e_p), counts[p]), 5)
        elo[o] = round(_decay(elo_o + K_FACTOR * (s2 - (1 - e_p)), counts[o]), 5)

        # --- Surface Elo update (skip rows with an unknown surface) ---
        if s in surf_elo:
            se_p, se_o = surf_elo[s][p], surf_elo[s][o]
            surface_elo_diff.append(round(se_p - se_o, 5))
            se = _expected(se_p, se_o)
            surf_counts[s][p] += 1
            surf_counts[s][o] += 1
            surf_elo[s][p] = round(_decay(se_p + K_FACTOR * (s1 - se), surf_counts[s][p]), 5)
            surf_elo[s][o] = round(_decay(se_o + K_FACTOR * (s2 - (1 - se)), surf_counts[s][o]), 5)
        else:
            surface_elo_diff.append(0.0)

    df["player_elo_before"] = player_elo_before
    df["opponent_elo_before"] = opponent_elo_before
    df["surface_elo_diff"] = surface_elo_diff

    # Persist final global ratings (handy for inference / inspection).
    elo_df = (
        pd.DataFrame(list(elo.items()), columns=["player_id", "elo"])
        .sort_values(by="elo", ascending=False)
        .reset_index(drop=True)
    )
    out_path = elo_output_path if elo_output_path is not None else PATHS["elo_ratings"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    elo_df.to_csv(out_path, index=False)

    return df
