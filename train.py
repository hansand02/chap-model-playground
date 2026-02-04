"""Train a simple linear regression model for disease prediction."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from visual import create_writer, log_loss


class MetaLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 32)
        self.layer2 = nn.Linear(32, 1)
        self.activation = nn.ReLU()
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 8)
        self.layer2 = nn.Linear(8, 8)
        self.layer3 = nn.Linear(8, 8)
        self.layer4 = nn.Linear(8, 1)
        self.activation = nn.ReLU()
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        x = self.activation(x)
        x = self.layer4(x)
        return x


class LinearRegression(nn.Module):
    def __init__(self, input_size=4):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.linear(x)


def evaluate(model, X, Y, model_name="Model"):
    """Evaluate the model and print metrics."""
    model.eval()
    with torch.no_grad():
        predictions = model(X)
        mse = ((predictions - Y) ** 2).mean().item()
        mae = (predictions - Y).abs().mean().item()
    print(f"{model_name} - MSE: {mse:.2f}, MAE: {mae:.2f}")
    return predictions


def plot_predictions(dates, actual, predictions_dict):
    """Plot actual vs predicted values for multiple models.

    Parameters
    ----------
    dates : array-like
        Date values for x-axis.
    actual : array-like
        Actual target values.
    predictions_dict : dict
        Dictionary mapping model names to their predictions.
    """
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


def train_model(model, X, Y, epochs=30000, lr=0.001, writer=None):
    """Train a model using Adam optimizer.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    X : torch.Tensor
        Input features.
    Y : torch.Tensor
        Target values.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    writer : SummaryWriter, optional
        TensorBoard writer for logging.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predict = model(X)
        loss = model.loss(predict, Y)
        if writer:
            log_loss(writer, loss.item(), epoch)
        loss.backward()
        optimizer.step()
    return model


def train(train_data_path, model_path):
    """Train both models on the provided data.

    Parameters
    ----------
    train_data_path : str
        Path to the training data CSV file.
    model_path : str
        Path where the trained model will be saved.
    """
    df = pd.read_csv(train_data_path)

    # Temporal train/validation split: train on 1998-2008, validate on 2009-2010
    train_df = df[df["time_period"] < "2009-01"]
    val_df = df[df["time_period"] >= "2009-01"]
    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

    feature_cols = [
        "population",
        "t2m_mean_c_month",
        "tp_sum_mm_month",
        "rh2m_mean_month",
    ]

    # Prepare training data
    train_inputs = train_df[feature_cols].fillna(0)
    train_mean, train_std = train_inputs.mean(), train_inputs.std()
    train_inputs = (train_inputs - train_mean) / train_std
    train_target = train_df["disease_cases"].fillna(0)

    # Prepare validation data (normalize with training stats to avoid leakage)
    val_inputs = val_df[feature_cols].fillna(0)
    val_inputs = (val_inputs - train_mean) / train_std
    val_target = val_df["disease_cases"].fillna(0)

    X_train = torch.tensor(train_inputs.values, dtype=torch.float32)
    Y_train = torch.tensor(train_target.values, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(val_inputs.values, dtype=torch.float32)
    Y_val = torch.tensor(val_target.values, dtype=torch.float32).unsqueeze(1)

    # Train Neural Network
    print("\nTraining Neural Network...")
    nn_model = SimpleNN()
    writer_nn = create_writer("disease_model_nn")
    train_model(nn_model, X_train, Y_train, epochs=3000, writer=writer_nn)
    writer_nn.close()
    print("Train metrics:")
    nn_train_preds = evaluate(nn_model, X_train, Y_train, "  Neural Network")
    print("Validation metrics:")
    nn_val_preds = evaluate(nn_model, X_val, Y_val, "  Neural Network")

    # Train Linear Regression
    print("\nTraining Linear Regression...")
    linear_model = LinearRegression(input_size=4)
    writer_linear = create_writer("disease_model_linear")
    train_model(linear_model, X_train, Y_train, epochs=3000, writer=writer_linear)
    writer_linear.close()
    print("Train metrics:")
    linear_train_preds = evaluate(linear_model, X_train, Y_train, "  Linear Regression")
    print("Validation metrics:")
    linear_val_preds = evaluate(linear_model, X_val, Y_val, "  Linear Regression")

    # Train Meta Learner on training predictions
    train_pred_tensor = torch.cat([nn_train_preds, linear_train_preds], dim=1)
    val_pred_tensor = torch.cat([nn_val_preds, linear_val_preds], dim=1)

    print("\nTraining Meta Model...")
    meta_learner = MetaLearner()
    writer_meta = create_writer("disease_model_meta")
    train_model(
        meta_learner, train_pred_tensor, Y_train, epochs=3000, writer=writer_meta
    )
    writer_meta.close()
    print("Train metrics:")
    evaluate(meta_learner, train_pred_tensor, Y_train, "  Meta Model")
    print("Validation metrics:")
    meta_val_preds = evaluate(meta_learner, val_pred_tensor, Y_val, "  Meta Model")

    # Aggregate predictions by time period (average across all regions)
    val_results = val_df[["time_period"]].copy()
    val_results["actual"] = Y_val.squeeze().tolist()
    val_results["nn_pred"] = nn_val_preds.squeeze().tolist()
    val_results["linear_pred"] = linear_val_preds.squeeze().tolist()
    val_results["meta_pred"] = meta_val_preds.squeeze().tolist()

    val_agg = val_results.groupby("time_period").mean().reset_index()
    val_dates = pd.to_datetime(val_agg["time_period"])

    predictions_dict = {
        "Neural Network": val_agg["nn_pred"].tolist(),
        "Linear Regression": val_agg["linear_pred"].tolist(),
        "Meta Model": val_agg["meta_pred"].tolist(),
    }
    plot_predictions(val_dates, val_agg["actual"].tolist(), predictions_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a disease prediction model")
    parser.add_argument("train_data", help="Path to training data CSV file")
    parser.add_argument("model", help="Path to save the trained model")
    args = parser.parse_args()

    train(args.train_data, args.model)
