import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Step 1: Load data
data = pd.read_csv('data/processed/all_matches.csv')

# Step 2: Preprocess
data = data.dropna()
le_surface = LabelEncoder()
le_tourney_level = LabelEncoder()
le_round = LabelEncoder()
data['surface'] = le_surface.fit_transform(data['surface'])
data['tourney_level'] = le_tourney_level.fit_transform(data['tourney_level'])
data['round'] = le_round.fit_transform(data['round'])
feature_columns = [
    'surface', 'tourney_level', 'best_of', 'round', 'rank_diff', 'points_diff',
    'age_diff', 'h2h_diff', 'player_elo_before', 'opponent_elo_before'
]
scaler = StandardScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])
target = data['win_loss'].values

# Step 3: Create sequences
seq_length = 10
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for player_id in data['player_id'].unique():
        player_data = data[data['player_id'] == player_id]
        features = player_data[feature_columns].values
        outcomes = player_data['win_loss'].values
        for i in range(len(player_data) - seq_length):
            seq = features[i:i + seq_length]
            target = outcomes[i + seq_length]
            sequences.append(seq)
            targets.append(target)
    return np.array(sequences), np.array(targets)

X, y = create_sequences(data, seq_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Step 4: Define dataset
class TennisDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TennisDataset(X_train, y_train)
test_dataset = TennisDataset(X_test, y_test)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 5: Define model
class TennisLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TennisLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

input_size = len(feature_columns)
hidden_size = 64
num_layers = 2
output_size = 1
model = TennisLSTM(input_size, hidden_size, num_layers, output_size)

# Step 6: Train model
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

# Step 7: Evaluate model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            predicted = (outputs >= 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

evaluate_model(model, test_loader)
