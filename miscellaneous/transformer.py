import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

# ---------------------------
# 1) DATA LOADING & PREPROCESS
# ---------------------------
print("Loading and preprocessing data...")
data = pd.read_csv('data/processed/all_matches.csv')
data = data.dropna()

# encode categoricals
le_surface      = LabelEncoder().fit(data['surface'])
le_tourney      = LabelEncoder().fit(data['tourney_level'])
le_round        = LabelEncoder().fit(data['round'])
data['surface']       = le_surface.transform(data['surface'])
data['tourney_level'] = le_tourney.transform(data['tourney_level'])
data['round']         = le_round.transform(data['round'])

# define separate feature sets
player_features = [
    'surface', 'tourney_level', 'best_of', 'round',
    'player_age', 'player_rank', 'player_rank_points',
    'player_elo_before', 'player_surface_elo_before'
]
opponent_features = [
    'surface', 'tourney_level', 'best_of', 'round',
    'opponent_age', 'opponent_rank', 'opponent_rank_points',
    'opponent_elo_before', 'opponent_surface_elo_before'
]

# scale continuous jointly so that both streams see the same scaling
scaler = StandardScaler()
data[player_features + opponent_features] = scaler.fit_transform(
    data[player_features + opponent_features]
)

# ---------------------------
# 2) SEQUENCE CREATION
# ---------------------------
seq_length = 10

def create_sequences(df, seq_length):
    Xp, Xo, ys = [], [], []
    # assume df is chronologically sorted
    for pid in df['player_id'].unique():
        sub = df[df['player_id'] == pid]
        pf = sub[player_features].values
        of = sub[opponent_features].values
        labels = sub['win_loss'].values
        for i in range(len(sub) - seq_length):
            Xp.append(pf[i:i+seq_length])
            Xo.append(of[i:i+seq_length])
            ys.append(labels[i+seq_length])
    return (
        np.stack(Xp), 
        np.stack(Xo), 
        np.array(ys, dtype=np.float32).reshape(-1, 1)
    )

Xp, Xo, y = create_sequences(data, seq_length)

# train/test split
Xp_train, Xp_test, Xo_train, Xo_test, y_train, y_test = train_test_split(
    Xp, Xo, y, test_size=0.2, random_state=42, stratify=y
)

# to torch
Xp_train = torch.tensor(Xp_train, dtype=torch.float32)
Xo_train = torch.tensor(Xo_train, dtype=torch.float32)
Xp_test  = torch.tensor(Xp_test,  dtype=torch.float32)
Xo_test  = torch.tensor(Xo_test,  dtype=torch.float32)
y_train  = torch.tensor(y_train,  dtype=torch.float32)
y_test   = torch.tensor(y_test,   dtype=torch.float32)

# ---------------------------
# 3) DATASET & DATALOADER
# ---------------------------
class TennisDataset(Dataset):
    def __init__(self, Xp, Xo, y):
        self.Xp, self.Xo, self.y = Xp, Xo, y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return (self.Xp[idx], self.Xo[idx], self.y[idx])

batch_size = 32
train_loader = DataLoader(
    TennisDataset(Xp_train, Xo_train, y_train),
    batch_size, shuffle=True
)
test_loader = DataLoader(
    TennisDataset(Xp_test, Xo_test, y_test),
    batch_size, shuffle=False
)

# ---------------------------
# 4) POSITIONAL ENCODING
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ---------------------------
# 5) DUAL-STREAM TRANSFORMER
# ---------------------------
class TennisTransformer(nn.Module):
    def __init__(self, dim_p, dim_o, d_model=64, nhead=4, num_layers=3,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        # project each stream into model-space
        self.input_proj_p = nn.Linear(dim_p, d_model)
        self.input_proj_o = nn.Linear(dim_o, d_model)

        self.pos_enc = PositionalEncoding(d_model, max_len=seq_length)

        # separate encoders
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_p = nn.TransformerEncoder(enc_layer, num_layers)
        self.transformer_o = nn.TransformerEncoder(enc_layer, num_layers)

        # classifier now takes 2*d_model
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, xp, xo):
        # xp, xo: [B, seq, dim_p] / [B, seq, dim_o]
        xp = self.input_proj_p(xp)
        xo = self.input_proj_o(xo)

        xp = self.pos_enc(xp)
        xo = self.pos_enc(xo)

        xp = self.transformer_p(xp)  # [B, seq, d_model]
        xo = self.transformer_o(xo)

        # take last time‐step
        hp = xp[:, -1, :]  # [B, d_model]
        ho = xo[:, -1, :]

        h  = torch.cat([hp, ho], dim=1)  # [B, 2*d_model]
        return self.classifier(h)

# ---------------------------
# 6) TRAIN & SAVE
# ---------------------------
print("Initializing model and training...")
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model      = TennisTransformer(
    dim_p=len(player_features),
    dim_o=len(opponent_features)
).to(device)
criterion  = nn.BCELoss()
optimizer  = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 30

for epoch in range(1, num_epochs+1):
    model.train()
    running_loss = 0.0
    for xp, xo, yb in train_loader:
        xp, xo, yb = xp.to(device), xo.to(device), yb.to(device)
        preds = model(xp, xo)
        loss  = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xp.size(0)

    print(f"[Epoch {epoch:02d}] Train Loss: {running_loss / len(train_loader.dataset):.4f}")

# persist everything
print("Saving model and preprocessors...")
joblib.dump(model, 'transformer_dual_stream.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(le_surface, 'le_surface.joblib')
joblib.dump(le_tourney, 'le_tourney.joblib')
joblib.dump(le_round,   'le_round.joblib')

# ---------------------------
# 7) EVALUATION
# ---------------------------
print("Evaluating on test set...")
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xp, xo, yb in test_loader:
        xp, xo, yb = xp.to(device), xo.to(device), yb.to(device)
        preds     = model(xp, xo)
        pred_lbl  = (preds >= 0.5).float()
        total    += yb.size(0)
        correct  += (pred_lbl == yb).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")
