import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, d_model=16, nhead=4, num_layers=2, seq_len=10):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.seq_len = seq_len

    def forward(self, x):
        # x: (seq_len, input_dim)
        x = self.embed(x) + self.pos_embed       # (seq_len, d_model)
        x = self.transformer(x)                  # (seq_len, d_model)
        return x.mean(dim=0)                     # (d_model,)


def generate_stats_with_transformer(df: pd.DataFrame, window: int = 10, lookback: int = 600) -> pd.DataFrame:
    
    device = torch.device('cpu')
    input_dim = 3               # [ace, df, bpSaved]
    d_model = 16
    encoder = SequenceEncoder(input_dim, d_model=d_model, seq_len=window).to(device)
    encoder.eval()

    p0_embeddings = []
    p1_embeddings = []

    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing row {idx}/{len(df)}...")
        start_idx = max(0, idx - lookback)
        prev_df = df.iloc[start_idx:idx]

        def get_sequence(entity_id):
            matches = prev_df[(prev_df['player_id'] == entity_id) | (prev_df['opponent_id'] == entity_id)]
            tail = matches.tail(window)
            seq = []
            for _, m in tail.iterrows():
                if m['player_id'] == entity_id:
                    seq.append([m['w_ace'], m['w_df'], m['w_bpSaved']])
                else:
                    seq.append([m['l_ace'], m['l_df'], m['l_bpSaved']])
            while len(seq) < window:
                seq.insert(0, [0.0, 0.0, 0.0])
            return torch.tensor(seq, dtype=torch.float32, device=device)

        seq_p0 = get_sequence(row['player_id'])
        seq_p1 = get_sequence(row['opponent_id'])

        with torch.no_grad():
            emb_p0 = encoder(seq_p0).cpu().numpy()
            emb_p1 = encoder(seq_p1).cpu().numpy()

        p0_embeddings.append(emb_p0)
        p1_embeddings.append(emb_p1)

    emb_dim = d_model
    emb_cols_p0 = [f'p0_seq_emb_{i}' for i in range(emb_dim)]
    emb_cols_p1 = [f'p1_seq_emb_{i}' for i in range(emb_dim)]
    emb_df = pd.DataFrame(p0_embeddings, columns=emb_cols_p0, index=df.index)
    emb_df = emb_df.join(pd.DataFrame(p1_embeddings, columns=emb_cols_p1, index=df.index))

    result_df = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
    return result_df
