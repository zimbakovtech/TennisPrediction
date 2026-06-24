#!/usr/bin/env python3
import os
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from functions.utils import read_file

# -------------------------
# Deterministic configuration
# -------------------------
SEED = 42

# =============================
# Data Loading and Preparation
# =============================

MODEL_DIR = os.path.join("src", "models", "joblib")
os.makedirs(MODEL_DIR, exist_ok=True)

# =============================
# Sequence Building (10 matches)
# =============================

PLAYER_TIME_VARYING_FEATURES = [
    'player_ace',
    'player_df',
    'player_bp',
    'player_elo_before',
    # 'player_surface_elo_before',
]

OPP_TIME_VARYING_FEATURES = [
    'opponent_ace',
    'opponent_df',
    'opponent_bp',
    'opponent_elo_before',
    # 'opponent_surface_elo_before',
]

STATIC_CONTEXT_FEATURES = [
    'best_of',
    'match_importance',
    'rank_diff',
    'points_diff',
    'age_diff',
    'h2h_diff',
]


@dataclass
class SequenceExample:
    player_seq: np.ndarray
    opponent_seq: np.ndarray
    static_context: np.ndarray
    label: int


def build_sequences(df: pd.DataFrame, seq_len: int = 10) -> List[SequenceExample]:
    df = df.copy()
    # No scaling here; moved to after splitting

    # Fill missing with 0
    scale_cols = set(PLAYER_TIME_VARYING_FEATURES + OPP_TIME_VARYING_FEATURES + STATIC_CONTEXT_FEATURES)
    for c in scale_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    examples: List[SequenceExample] = []
    # Build per-entity chronological indices to fetch last N matches
    # We'll use the already time-sorted df

    # For each row (a match), construct sequences for player and opponent histories
    for idx in range(len(df)):
        row = df.iloc[idx]
        pid = row["player_id"]
        oid = row["opponent_id"]
        # Prior matches for player and opponent up to but excluding idx
        history_slice = df.iloc[max(0, idx - 2000):idx]
        p_hist = history_slice[(history_slice["player_id"] == pid)]
        o_hist = history_slice[(history_slice["player_id"] == oid)]

        def seq_from_hist(hist: pd.DataFrame, tv_cols: List[str]) -> np.ndarray:
            if hist.empty:
                return np.zeros((seq_len, len(tv_cols)), dtype=np.float32)
            # Take the last seq_len rows of history
            tail = hist.tail(seq_len)
            arr = tail[tv_cols].to_numpy(dtype=np.float32)
            if len(arr) < seq_len:
                pad = np.zeros((seq_len - len(arr), arr.shape[1]), dtype=np.float32)
                arr = np.vstack([pad, arr])
            return arr

        p_seq = seq_from_hist(p_hist, PLAYER_TIME_VARYING_FEATURES)
        o_seq = seq_from_hist(o_hist, OPP_TIME_VARYING_FEATURES)

        static_vec = row[STATIC_CONTEXT_FEATURES].to_numpy(dtype=float).astype(np.float32)
        label = int(row["win_loss"])  

        examples.append(SequenceExample(player_seq=p_seq, opponent_seq=o_seq, static_context=static_vec, label=label))

    # print(f"Last 10: {examples[-10:]}")
    # print(f"First 10: {examples[:10]}")
    # print(examples[26445:26455])
    return examples


# =============================
# Dataset / Dataloader
# =============================


class TennisSeqDataset(Dataset):
    def __init__(self, examples: List[SequenceExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        return (
            torch.from_numpy(ex.player_seq),
            torch.from_numpy(ex.opponent_seq),
            torch.from_numpy(ex.static_context),
            torch.tensor(ex.label, dtype=torch.float32),
        )


# =============================
# Simplified Temporal Fusion Model
# =============================


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        # GLU expects 2*channels input; we will concatenate residual accordingly when calling
        self.gate = nn.GLU()
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.layernorm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.lin1(x)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        # Gate: concatenate x and residual along last dim (so GLU halves channels)
        # ensure dims align: x has output_dim, residual has output_dim, concat -> 2*output_dim -> GLU -> output_dim
        gated = self.gate(torch.cat([x, residual], dim=-1))
        return self.layernorm(residual + gated)


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, num_heads: int = 4, dropout: float = 0.1, max_len: int = 32):
        super().__init__()
        self.value_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads,
                                                   dim_feedforward=model_dim * 2, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.temporal_gate = GatedResidualNetwork(model_dim, model_dim * 2, model_dim, dropout=dropout)
        # Learned positional embeddings
        self.pos_emb = nn.Embedding(max_len, model_dim)
        self.max_len = max_len

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, T, F]
        B, T, _ = x_seq.shape
        x = self.value_proj(x_seq)  # [B, T, model_dim]
        if T > self.max_len:
            pos = torch.arange(T, device=x.device).unsqueeze(0)
            pos_emb = self.pos_emb(pos % self.max_len)
        else:
            pos = torch.arange(T, device=x.device).unsqueeze(0)
            pos_emb = self.pos_emb(pos)  # [1, T, model_dim]
        x = x + pos_emb  # broadcast over B
        x = self.encoder(x)
        x = self.temporal_gate(x)
        pooled = x.mean(dim=1)
        return x, pooled

class TFTLike(nn.Module):
    def __init__(self, player_feat_dim: int, opp_feat_dim: int, static_dim: int, model_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.player_encoder = TemporalEncoder(player_feat_dim, model_dim, num_heads=4, dropout=dropout)
        self.opp_encoder = TemporalEncoder(opp_feat_dim, model_dim, num_heads=4, dropout=dropout)

        self.static_processor = GatedResidualNetwork(static_dim, model_dim * 2, model_dim, dropout=dropout)

        self.fusion_gate = GatedResidualNetwork(model_dim * 3, model_dim * 2, model_dim, dropout=dropout)
        self.comparator = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

    def forward(self, p_seq: torch.Tensor, o_seq: torch.Tensor, static_ctx: torch.Tensor) -> torch.Tensor:
        # Encode sequences
        _, p_vec = self.player_encoder(p_seq)
        _, o_vec = self.opp_encoder(o_seq)
        s_vec = self.static_processor(static_ctx)

        # Compare player vs opponent representation with static context
        fused = torch.cat([p_vec, o_vec, s_vec], dim=-1)
        fused = self.fusion_gate(fused)
        logits = self.comparator(fused).squeeze(-1)
        return logits


# =============================
# Training / Evaluation
# =============================


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 5, lr: float = 1e-3, device: str = "cpu") -> None:
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for p_seq, o_seq, s_ctx, y in train_loader:
            p_seq, o_seq, s_ctx, y = p_seq.to(device), o_seq.to(device), s_ctx.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(p_seq, o_seq, s_ctx)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
        avg_loss = total_loss / max(1, len(train_loader.dataset))

        # Validation
        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for p_seq, o_seq, s_ctx, y in val_loader:
                p_seq, o_seq, s_ctx, y = p_seq.to(device), o_seq.to(device), s_ctx.to(device), y.to(device)
                logits = model(p_seq, o_seq, s_ctx)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct += (preds == y).sum().item()
                total += y.numel()
            val_acc = correct / max(1, total)
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f} val_acc={val_acc:.4f}")


def evaluate_and_save(model: nn.Module, loader: DataLoader, device: str = "cpu", out_csv: Optional[str] = None) -> float:
    model.eval()
    correct, total = 0, 0
    rows = []
    with torch.no_grad():
        for p_seq, o_seq, s_ctx, y in loader:
            p_seq, o_seq, s_ctx, y = p_seq.to(device), o_seq.to(device), s_ctx.to(device), y.to(device)
            logits = model(p_seq, o_seq, s_ctx)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(np.float32)
            correct += (preds == y.cpu().numpy()).sum()
            total += y.numel()
            for pr, gt in zip(probs, y.cpu().numpy().tolist()):
                rows.append({"pred_prob_player_win": float(pr), "label": int(gt)})
    acc = correct / max(1, total)
    print(f"Test accuracy: {acc:.4f}")
    if out_csv is not None:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved predictions to {out_csv}")
    return acc


def split_examples(examples: List[SequenceExample], val_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[List[SequenceExample], List[SequenceExample], List[SequenceExample]]:
    # Time-based deterministic split (keeps chronological ordering)
    n = len(examples)
    n_test = int(n * test_ratio)
    n_val = int((n - n_test) * val_ratio)
    test = examples[-n_test:] if n_test > 0 else []
    trainval = examples[:-n_test] if n_test > 0 else examples
    val = trainval[-n_val:] if n_val > 0 else []
    train = trainval[:-n_val] if n_val > 0 else trainval
    return train, val, test


def mirror_example(ex: SequenceExample, flip_idx: List[int]) -> SequenceExample:
    # create mirrored example by swapping player/opponent sequences and flipping appropriate static indices and label
    p_seq = ex.opponent_seq.copy()
    o_seq = ex.player_seq.copy()
    s_ctx = ex.static_context.copy()
    # Flip the selected indices (if in range)
    for i in flip_idx:
        if 0 <= i < len(s_ctx):
            s_ctx[i] = -s_ctx[i]
    label = 1 - int(ex.label)
    return SequenceExample(player_seq=p_seq, opponent_seq=o_seq, static_context=s_ctx, label=label)


def mirror_examples_list(ex_list: List[SequenceExample], flip_idx: List[int]) -> List[SequenceExample]:
    out = []
    for ex in ex_list:
        out.append(ex)
        out.append(mirror_example(ex, flip_idx))
    return out


def main():
    # ------------------------------
    # Set deterministic seeds
    # ------------------------------
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # cuDNN deterministic settings (may slow things down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df, _ = read_file()

    print("Building sequences (10 matches per player/opponent)...")
    examples = build_sequences(df, seq_len=10)
    print(f"Raw sequences: {len(examples)}")

    # Time-based split on original examples (deterministic)
    train_orig, val_ex, test_ex = split_examples(examples, val_ratio=0.1, test_ratio=0.1)
    print(f"Train_orig: {len(train_orig)}, Val: {len(val_ex)}, Test: {len(test_ex)}")

    flip_feats = ["rank_diff", "points_diff", "age_diff", "h2h_diff"]
    flip_idx = [STATIC_CONTEXT_FEATURES.index(f) for f in flip_feats if f in STATIC_CONTEXT_FEATURES]

    train_ex = mirror_examples_list(train_orig, flip_idx)
    val_ex = mirror_examples_list(val_ex, flip_idx)
    test_ex = mirror_examples_list(test_ex, flip_idx)

    # Optionally shuffle only the training examples (deterministic)
    # rng = random.Random(SEED)
    # rng.shuffle(train_orig)

    # Print label counts per split to confirm balance
    def label_counts(lst: List[SequenceExample]):
        labels = [ex.label for ex in lst]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    print("Label counts -> train:", label_counts(train_ex), "val:", label_counts(val_ex), "test:", label_counts(test_ex))

    # Compute robust scaling parameters from train only (train_ex contains both classes now)
    def collect_values(ex_list: List[SequenceExample]):
        p_vals = [[] for _ in range(len(PLAYER_TIME_VARYING_FEATURES))]
        o_vals = [[] for _ in range(len(OPP_TIME_VARYING_FEATURES))]
        s_vals = [[] for _ in range(len(STATIC_CONTEXT_FEATURES))]
        for ex in ex_list:
            for fi in range(len(PLAYER_TIME_VARYING_FEATURES)):
                p_vals[fi].extend(ex.player_seq[:, fi].tolist())
            for fi in range(len(OPP_TIME_VARYING_FEATURES)):
                o_vals[fi].extend(ex.opponent_seq[:, fi].tolist())
            for fi in range(len(STATIC_CONTEXT_FEATURES)):
                s_vals[fi].append(ex.static_context[fi])
        return p_vals, o_vals, s_vals

    p_vals, o_vals, s_vals = collect_values(train_ex)

    player_meds = np.array([np.median(v) if len(v) > 0 else 0.0 for v in p_vals], dtype=np.float32)
    player_mads = np.array([np.median(np.abs(np.array(v, dtype=np.float32) - m)) if len(v) > 0 else 1.0 for v, m in zip(p_vals, player_meds)], dtype=np.float32)
    player_mads[player_mads == 0] = 1.0

    opp_meds = np.array([np.median(v) if len(v) > 0 else 0.0 for v in o_vals], dtype=np.float32)
    opp_mads = np.array([np.median(np.abs(np.array(v, dtype=np.float32) - m)) if len(v) > 0 else 1.0 for v, m in zip(o_vals, opp_meds)], dtype=np.float32)
    opp_mads[opp_mads == 0] = 1.0

    static_meds = np.array([np.median(v) if len(v) > 0 else 0.0 for v in s_vals], dtype=np.float32)
    static_mads = np.array([np.median(np.abs(np.array(v, dtype=np.float32) - m)) if len(v) > 0 else 1.0 for v, m in zip(s_vals, static_meds)], dtype=np.float32)
    static_mads[static_mads == 0] = 1.0

    # Apply scaling to all splits
    def apply_scale(ex_list: List[SequenceExample]):
        for ex in ex_list:
            # broadcast subtract along feature dimension
            ex.player_seq = (ex.player_seq - player_meds) / player_mads
            ex.opponent_seq = (ex.opponent_seq - opp_meds) / opp_mads
            ex.static_context = (ex.static_context - static_meds) / static_mads

    # apply_scale(train_ex)
    # apply_scale(val_ex)
    # apply_scale(test_ex)

    train_ds, val_ds, test_ds = TennisSeqDataset(train_ex), TennisSeqDataset(val_ex), TennisSeqDataset(test_ex)

    # Deterministic DataLoader generator for shuffle
    # train_gen = torch.Generator()
    # train_gen.manual_seed(SEED)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    # Seed again before model instantiation to ensure deterministic weight init
    torch.manual_seed(SEED)
    model = TFTLike(
        player_feat_dim=len(PLAYER_TIME_VARYING_FEATURES),
        opp_feat_dim=len(OPP_TIME_VARYING_FEATURES),
        static_dim=len(STATIC_CONTEXT_FEATURES),
        model_dim=64,
        dropout=0.1,
    )

    print("Training model...")
    # use a smaller lr for stability
    train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device=device)

    print("Evaluating and saving artifacts...")
    acc = evaluate_and_save(model, test_loader, device=device, out_csv=os.path.join(MODEL_DIR, "tft_predictions.csv"))
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "tft_model.pth"))
    print(f"Saved model to {os.path.join(MODEL_DIR, 'tft_model.pth')} (test_acc={acc:.4f})")


if __name__ == "__main__":
    main()
