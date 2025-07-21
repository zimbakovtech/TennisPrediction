import math
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import joblib
import torch.nn as nn
from sklearn.model_selection import train_test_split

# ---------------------------
# Paths and Constants
# ---------------------------
PREPROCESS_PATHS = {
    'surface': 'le_surface.joblib',
    'tourney_level': 'le_tourney.joblib',
    'round': 'le_round.joblib',
    'scaler': 'scaler.joblib',
}
MODEL_PATH = 'transformer_dual_stream.joblib'
DATA_PATH = 'data/processed/all_matches.csv'
PLAYER_FEATURES = [
    'surface', 'tourney_level', 'best_of', 'round',
    'player_age', 'player_rank', 'player_rank_points',
    'player_elo_before', 'player_surface_elo_before'
]
OPPONENT_FEATURES = [
    'surface', 'tourney_level', 'best_of', 'round',
    'opponent_age', 'opponent_rank', 'opponent_rank_points',
    'opponent_elo_before', 'opponent_surface_elo_before'
]
SEQ_LENGTH = 10
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------
# Model & Dataset Definitions
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class TennisTransformer(nn.Module):
    def __init__(self, dim_p, dim_o, d_model=64, nhead=4, num_layers=3,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj_p = nn.Linear(dim_p, d_model)
        self.input_proj_o = nn.Linear(dim_o, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_p = nn.TransformerEncoder(enc_layer, num_layers)
        self.transformer_o = nn.TransformerEncoder(enc_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, xp: torch.Tensor, xo: torch.Tensor) -> torch.Tensor:
        xp = self.pos_enc(self.input_proj_p(xp))
        xo = self.pos_enc(self.input_proj_o(xo))
        xp = self.transformer_p(xp)
        xo = self.transformer_o(xo)
        out = torch.cat([xp[:, -1, :], xo[:, -1, :]], dim=1)
        return self.classifier(out)

class TennisDataset(torch.utils.data.Dataset):
    def __init__(self, Xp, Xo, y):
        self.Xp = Xp
        self.Xo = Xo
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.Xp[idx], self.Xo[idx], self.y[idx]

# ---------------------------
# Utility Functions
# ---------------------------
def load_preprocessors(paths):
    return {name: joblib.load(path) for name, path in paths.items()}

def preprocess_data(path, encoders, scaler):
    df = pd.read_csv(path)
    for name, le in encoders.items():
        df[name] = le.transform(df[name])
    df[PLAYER_FEATURES + OPPONENT_FEATURES] = scaler.transform(df[PLAYER_FEATURES + OPPONENT_FEATURES])
    return df

def create_sequences(df, seq_length):
    Xp, Xo, ys = [], [], []
    for pid in df['player_id'].unique():
        sub = df[df['player_id'] == pid]
        pf = sub[PLAYER_FEATURES].values
        of = sub[OPPONENT_FEATURES].values
        labels = sub['win_loss'].values
        for i in range(len(sub) - seq_length):
            Xp.append(pf[i:i + seq_length])
            Xo.append(of[i:i + seq_length])
            ys.append(labels[i + seq_length])
    return np.stack(Xp), np.stack(Xo), np.array(ys, dtype=np.float32).reshape(-1, 1)

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xp, xo, yb in loader:
            xp, xo, yb = xp.to(device), xo.to(device), yb.to(device)
            preds = model(xp, xo)
            predicted = (preds >= 0.5).float()
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
    return correct / total

# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading model and preprocessors...")
    preprocessors = load_preprocessors(PREPROCESS_PATHS)
    scaler = preprocessors.pop('scaler')
    encoders = preprocessors
    model: TennisTransformer = joblib.load(MODEL_PATH)
    model.to(device)

    print("Loading and preprocessing data...")
    df = preprocess_data(DATA_PATH, encoders, scaler)

    print("Creating sequences...")
    Xp, Xo, y = create_sequences(df, SEQ_LENGTH)
    Xp_train, Xp_test, Xo_train, Xo_test, y_train, y_test = train_test_split(
        Xp, Xo, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    Xp_train = torch.tensor(Xp_train, dtype=torch.float32)
    Xo_train = torch.tensor(Xo_train, dtype=torch.float32)
    y_train  = torch.tensor(y_train,  dtype=torch.float32)
    Xp_test  = torch.tensor(Xp_test,  dtype=torch.float32)
    Xo_test  = torch.tensor(Xo_test,  dtype=torch.float32)
    y_test   = torch.tensor(y_test,   dtype=torch.float32)

    train_loader = DataLoader(TennisDataset(Xp_train, Xo_train, y_train), BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TennisDataset(Xp_test,  Xo_test,  y_test),  BATCH_SIZE, shuffle=False)

    print("Evaluating on training set...")
    train_acc = evaluate(model, train_loader, device)
    print(f"Train Accuracy: {train_acc * 100:.2f}%")

    print("Evaluating on test set...")
    test_acc = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    print("Predicting for the last 127 matches...")
    last_indices = df.index[-127:]
    player_matches = df.groupby('player_id').apply(lambda x: x.index.tolist()).to_dict()
    xp_seqs, xo_seqs, metadata = [], [], []

    for idx in last_indices:
        pid = df.loc[idx, 'player_id']
        matches = player_matches.get(pid, [])
        if idx in matches:
            k = matches.index(idx)
            if k >= SEQ_LENGTH:
                seq_idx = matches[k - SEQ_LENGTH:k]
                xp = df.loc[seq_idx, PLAYER_FEATURES].values
                xo = df.loc[seq_idx, OPPONENT_FEATURES].values
                xp_seqs.append(xp)
                xo_seqs.append(xo)
                metadata.append((pid, idx))
            else:
                print(f"Not enough history for player {pid} at match {idx}.")

    if xp_seqs:
        xp_tensor = torch.tensor(np.stack(xp_seqs), dtype=torch.float32).to(device)
        xo_tensor = torch.tensor(np.stack(xo_seqs), dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            preds = model(xp_tensor, xo_tensor)
            preds = (preds >= 0.5).float().cpu().numpy()
        for (pid, idx), pred in zip(metadata, preds):
            print(f"Match idx {idx} – Player ID: {pid} – Prediction: {int(pred[0])}")
    else:
        print("No sequences available for prediction.")
    
    print(preds.flatten().astype(int).tolist())
