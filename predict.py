"""Generate predictions using a trained model provider."""

import argparse

import pandas as pd
import torch
from model_provider import load_provider

FEATURE_COLS = [
    "rainfall",
    "mean_temperature",
    "population",
]


def predict(model_path, historic_data_path, future_data_path, out_file_path):
    """Generate predictions using a trained model provider.

    Parameters
    ----------
    model_path : str
        Path to the trained model file.
    historic_data_path : str
        Path to historic data CSV file (unused in this simple model).
    future_data_path : str
        Path to future climate data CSV file.
    out_file_path : str
        Path where predictions will be saved.
    """
    provider = load_provider(model_path)

    # Load normalization stats saved during training
    stats_path = model_path + ".stats"
    stats = torch.load(stats_path, weights_only=False)
    train_mean = pd.Series(stats["mean"])
    train_std = pd.Series(stats["std"])

    future_df = pd.read_csv(future_data_path)
    features = future_df[FEATURE_COLS].fillna(0)
    features = (features - train_mean) / train_std

    X = torch.tensor(features.values, dtype=torch.float32)
    predictions = provider.predict(X)

    output_df = future_df[["time_period", "location"]].copy()
    output_df["sample_0"] = predictions.squeeze().tolist()
    output_df.to_csv(out_file_path, index=False)
    print(f"Predictions saved to {out_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate disease predictions")
    parser.add_argument("model", help="Path to trained model file")
    parser.add_argument("historic_data", help="Path to historic data CSV file")
    parser.add_argument("future_data", help="Path to future climate data CSV file")
    parser.add_argument("out_file", help="Path to save predictions CSV file")
    args = parser.parse_args()

    predict(args.model, args.historic_data, args.future_data, args.out_file)
