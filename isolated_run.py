"""Run the model in isolation without CHAP integration.

This script demonstrates how to train and predict using the model directly,
which is useful for development and debugging before integrating with CHAP.
"""

import subprocess

# Train the model
subprocess.run(["uv", "run", "python", "main.py", "train", "input/trainData.csv", "output/model.pkl"], check=True)

# Generate predictions
subprocess.run(
    [
        "uv",
        "run",
        "python",
        "main.py",
        "predict",
        "output/model.pkl",
        "input/trainData.csv",
        "input/futureClimateData.csv",
        "output/predictions.csv",
    ],
    check=True,
)

print("\nPredictions saved to output/predictions.csv")
