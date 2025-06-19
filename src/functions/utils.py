import numpy as np
import pandas as pd


def process_y():
    df = pd.read_csv("data/processed/all_matches.csv")
    return df['win_loss'].values
    return np.array([1, 0] * (74906)).tolist()



def read_file():
    df = pd.read_csv("data/processed/all_matches.csv")
    df = df.drop(columns='win_loss')
    return df.values, df.columns