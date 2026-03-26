"""Run the model in isolation without CHAP integration.

This script demonstrates how to train and predict using the model directly,
which is useful for development and debugging before integrating with CHAP.
"""

from predict import predict
from train import train

# Train the model using the ensemble provider
train("input/trainData.csv", "output/model.pkl")

# Generate predictions
predict(
    "output/model.pkl",
    "input/trainData.csv",
    "input/futureClimateData.csv",
    "output/predictions.csv",
)

print("\nDone! Check the output/ directory for results.")
