"""Train a disease prediction model using a selectable provider."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import torch

from model_provider import PROVIDERS


FEATURE_COLS = [
    "rainfall",
    "mean_temperature",
    "population",
]


def plot_predictions(dates, actual, predictions_dict):
    """Plot actual vs predicted values for multiple models."""
    plt.figure(figsize=(10, 6))
    plt.plot(dates, actual, label="Actual", linewidth=2)
    for name, preds in predictions_dict.items():
        plt.plot(dates, preds, label=name, linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Disease Cases")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()


def train(train_data_path, model_path, provider_name="ensemble"):
    """Train a model on the provided data.

    Parameters
    ----------
    train_data_path : str
        Path to the training data CSV file.
    model_path : str
        Path where the trained model will be saved.
    provider_name : str
        Name of the model provider to use (linear, nn, ensemble).
    """
    df = pd.read_csv(train_data_path)

    # Temporal train/validation split: train on 1998-2008, validate on 2009-2010
    train_df = df[df["time_period"] < "2009-01"]
    val_df = df[df["time_period"] >= "2009-01"]
    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

    # Prepare training data
    train_inputs = train_df[FEATURE_COLS].fillna(0)
    train_mean, train_std = train_inputs.mean(), train_inputs.std()
    train_inputs = (train_inputs - train_mean) / train_std
    train_target = train_df["disease_cases"].fillna(0)

    # Prepare validation data (normalize with training stats to avoid leakage)
    val_inputs = val_df[FEATURE_COLS].fillna(0)
    val_inputs = (val_inputs - train_mean) / train_std
    val_target = val_df["disease_cases"].fillna(0)

    X_train = torch.tensor(train_inputs.values, dtype=torch.float32)
    Y_train = torch.tensor(train_target.values, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(val_inputs.values, dtype=torch.float32)
    Y_val = torch.tensor(val_target.values, dtype=torch.float32).unsqueeze(1)

    # Train using the selected provider
    provider_cls = PROVIDERS[provider_name]
    provider = provider_cls()

    print(f"\nTraining with provider: {provider_name}")
    provider.train(X_train, Y_train)

    print("\nTrain metrics:")
    provider.evaluate(X_train, Y_train, f"  {provider_name}")
    print("Validation metrics:")
    val_preds = provider.evaluate(X_val, Y_val, f"  {provider_name}")

    # Save model + normalization stats
    provider.save(model_path)
    stats_path = model_path + ".stats"
    torch.save({"mean": train_mean.to_dict(), "std": train_std.to_dict()}, stats_path)
    print(f"\nModel saved to {model_path}")

    # Plot validation predictions
    val_results = val_df[["time_period"]].copy()
    val_results["actual"] = Y_val.squeeze().tolist()
    val_results["predicted"] = val_preds.squeeze().tolist()

    val_agg = val_results.groupby("time_period").mean().reset_index()
    val_dates = pd.to_datetime(val_agg["time_period"])

    plot_predictions(
        val_dates,
        val_agg["actual"].tolist(),
        {provider_name: val_agg["predicted"].tolist()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a disease prediction model")
    parser.add_argument("train_data", help="Path to training data CSV file")
    parser.add_argument("model", help="Path to save the trained model")
    parser.add_argument(
        "--provider",
        default="ensemble",
        choices=list(PROVIDERS.keys()),
        help="Model provider to use (default: ensemble)",
    )
    args = parser.parse_args()

    train(args.train_data, args.model, args.provider)
