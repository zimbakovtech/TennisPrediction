import math
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import joblib
import torch.nn as nn

# ---------------------------
# MODEL DEFINITION (for loading weights)
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class TennisTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=100)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier  = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.classifier(x)

# ---------------------------
# 1) LOAD MODEL & PREPROCESSORS
# ---------------------------
print("Loading model and preprocessing objects...")
model      = joblib.load('transformer_model.joblib')
scaler     = joblib.load('scaler.joblib')
le_surface = joblib.load('le_surface.joblib')
le_tourney = joblib.load('le_tourney.joblib')
le_round   = joblib.load('le_round.joblib')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# ---------------------------
# 2) LOAD & PREPROCESS WIMBLEDON DATA
# ---------------------------
print("Loading Wimbledon 2025 data...")
data = pd.read_csv('data/testing/wimbledon_2025.csv')
data = data.dropna()

# encode categories
data['surface']      = le_surface.transform(data['surface'])
data['tourney_level']= le_tourney.transform(data['tourney_level'])
data['round']        = le_round.transform(data['round'])

feature_columns = [
    'surface', 'tourney_level', 'best_of', 'round',
    'rank_diff', 'points_diff', 'age_diff',
    'h2h_diff', 'player_elo_before', 'opponent_elo_before'
]
# scale
data[feature_columns] = scaler.transform(data[feature_columns])

# ---------------------------
# 3) CREATE SEQUENCES
# ---------------------------
seq_length = 10

def create_sequences(df, seq_length):
    Xs, ys = [], []
    for pid in df['player_id'].unique():
        sub = df[df['player_id'] == pid].sort_values('date')
        feats  = sub[feature_columns].values
        labels = sub['win_loss'].values
        if len(sub) <= seq_length:
            continue
        for i in range(len(sub) - seq_length):
            Xs.append(feats[i:i+seq_length])
            ys.append(labels[i+seq_length])
    return np.stack(Xs), np.array(ys)

X, y = create_sequences(data, seq_length)

# Convert to tensors
tensor_X = torch.tensor(X, dtype=torch.float32)
tensor_y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# DataLoader
class TennisDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

loader = DataLoader(TennisDataset(tensor_X, tensor_y), batch_size=32, shuffle=False)

# ---------------------------
# 4) PREDICT & EVALUATE
# ---------------------------
print("Predicting and evaluating Wimbledon matches...")
correct, total = 0, 0
all_preds = []
with torch.no_grad():
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        preds = model(Xb)
        predicted = (preds >= 0.5).float()
        total += yb.size(0)
        correct += (predicted == yb).sum().item()
        all_preds.extend(predicted.cpu().numpy().flatten().tolist())

accuracy = correct / total
print(f"Wimbledon 2025 Accuracy: {accuracy * 100:.2f}%")

# Optionally save predictions alongside true labels
out_df = pd.DataFrame({
    'true': tensor_y.numpy().flatten(),
    'pred': np.array(all_preds, dtype=int)
})
out_df.to_csv('wimbledon_2025_predictions.csv', index=False)
print("Saved predictions to wimbledon_2025_predictions.csv")
