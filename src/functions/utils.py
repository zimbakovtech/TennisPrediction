import numpy as np
import pandas as pd


def process_y():
    return np.array([1, 0] * 15710)


def read_file():
    df = pd.read_csv("data/processed/all_matches.csv")
    return df.values, df.columns
