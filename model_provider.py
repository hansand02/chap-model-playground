"""Model providers for disease prediction.

Each provider wraps a PyTorch model and handles training, prediction,
and serialization. Use the PROVIDERS registry to look up providers by name.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from visual import create_writer, log_loss

# ---------------------------------------------------------------------------
# PyTorch model definitions
# ---------------------------------------------------------------------------


class _SimpleNN(nn.Module):
    def __init__(self, input_size=3):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 8)
        self.layer4 = nn.Linear(8, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        x = self.activation(x)
        x = self.layer4(x)
        return x


class _LinearRegression(nn.Module):
    def __init__(self, input_size=3):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


class _MetaLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 32)
        self.layer2 = nn.Linear(32, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------


def _train_loop(model, X, Y, epochs=3000, lr=0.001, writer=None):
    """Train a model using Adam optimizer."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, Y)
        if writer:
            log_loss(writer, loss.item(), epoch)
        loss.backward()
        optimizer.step()


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------


class ModelProvider(ABC):
    """Base class for model providers."""

    name: str

    @abstractmethod
    def train(self, X_train, Y_train):
        """Train the model on the given data."""

    @abstractmethod
    def predict(self, X):
        """Return predictions as a torch.Tensor of shape (N, 1)."""

    @abstractmethod
    def save(self, path):
        """Persist the trained model to *path*."""

    @classmethod
    @abstractmethod
    def load(cls, path):
        """Load a previously saved model from *path* and return a provider."""

    def evaluate(self, X, Y, label="Model"):
        """Evaluate the model and print MSE / MAE."""
        preds = self.predict(X)
        mse = ((preds - Y) ** 2).mean().item()
        mae = (preds - Y).abs().mean().item()
        print(f"{label} - MSE: {mse:.2f}, MAE: {mae:.2f}")
        return preds


# ---------------------------------------------------------------------------
# Concrete providers
# ---------------------------------------------------------------------------


class LinearRegressionProvider(ModelProvider):
    name = "linear"

    def __init__(self):
        self.model = _LinearRegression(input_size=3)

    def train(self, X_train, Y_train):
        writer = create_writer("disease_model_linear")
        _train_loop(self.model, X_train, Y_train, epochs=3000, writer=writer)
        writer.close()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X)

    def save(self, path):
        torch.save({"provider": self.name, "model": self.model.state_dict()}, path)

    @classmethod
    def load(cls, path):
        provider = cls()
        data = torch.load(path, weights_only=False)
        provider.model.load_state_dict(data["model"])
        return provider


class SimpleNNProvider(ModelProvider):
    name = "nn"

    def __init__(self):
        self.model = _SimpleNN()

    def train(self, X_train, Y_train):
        writer = create_writer("disease_model_nn")
        _train_loop(self.model, X_train, Y_train, epochs=3000, writer=writer)
        writer.close()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X)

    def save(self, path):
        torch.save({"provider": self.name, "model": self.model.state_dict()}, path)

    @classmethod
    def load(cls, path):
        provider = cls()
        data = torch.load(path, weights_only=False)
        provider.model.load_state_dict(data["model"])
        return provider


class EnsembleProvider(ModelProvider):
    name = "ensemble"

    def __init__(self):
        self.nn_provider = SimpleNNProvider()
        self.linear_provider = LinearRegressionProvider()
        self.meta_learner = _MetaLearner()

    def train(self, X_train, Y_train):
        print("Training Neural Network...")
        self.nn_provider.train(X_train, Y_train)

        print("Training Linear Regression...")
        self.linear_provider.train(X_train, Y_train)

        # Build meta-learner inputs from base model predictions
        nn_preds = self.nn_provider.predict(X_train)
        linear_preds = self.linear_provider.predict(X_train)
        meta_input = torch.cat([nn_preds, linear_preds], dim=1)

        print("Training Meta Learner...")
        writer = create_writer("disease_model_meta")
        _train_loop(self.meta_learner, meta_input, Y_train, epochs=3000, writer=writer)
        writer.close()

    def predict(self, X):
        nn_preds = self.nn_provider.predict(X)
        linear_preds = self.linear_provider.predict(X)
        meta_input = torch.cat([nn_preds, linear_preds], dim=1)
        self.meta_learner.eval()
        with torch.no_grad():
            return self.meta_learner(meta_input)

    def save(self, path):
        torch.save(
            {
                "provider": self.name,
                "nn": self.nn_provider.model.state_dict(),
                "linear": self.linear_provider.model.state_dict(),
                "meta": self.meta_learner.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path):
        provider = cls()
        data = torch.load(path, weights_only=False)
        provider.nn_provider.model.load_state_dict(data["nn"])
        provider.linear_provider.model.load_state_dict(data["linear"])
        provider.meta_learner.load_state_dict(data["meta"])
        return provider


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, type[ModelProvider]] = {
    "linear": LinearRegressionProvider,
    "nn": SimpleNNProvider,
    "ensemble": EnsembleProvider,
}


def load_provider(path) -> ModelProvider:
    """Load a saved provider, auto-detecting its type."""
    data = torch.load(path, weights_only=False)
    provider_name = data["provider"]
    return PROVIDERS[provider_name].load(path)
