"""
Bonus: Compute precision metrics for the trained model against the training data.
Metrics: MAE, RMSE, and R² (coefficient of determination).
"""
import csv
import json
import math
import os
import sys

DATA_FILE = "data.csv"
MODEL_FILE = "model.json"


def load_data(filepath):
    km, prices = [], []
    try:
        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    km.append(float(row["km"]))
                    prices.append(float(row["price"]))
                except (ValueError, KeyError):
                    pass
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found.")
        sys.exit(1)
    return km, prices


def load_thetas():
    if not os.path.exists(MODEL_FILE):
        print(f"Error: no trained model found. Run train.py first.")
        sys.exit(1)
    try:
        with open(MODEL_FILE) as f:
            data = json.load(f)
        return float(data["theta0"]), float(data["theta1"])
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def main():
    km, prices = load_data(DATA_FILE)
    theta0, theta1 = load_thetas()
    m = len(km)

    predicted = [theta0 + theta1 * k for k in km]

    # Mean Absolute Error
    mae = sum(abs(predicted[i] - prices[i]) for i in range(m)) / m

    # Root Mean Squared Error
    mse = sum((predicted[i] - prices[i]) ** 2 for i in range(m)) / m
    rmse = math.sqrt(mse)

    # R² — coefficient of determination
    mean_price = sum(prices) / m
    ss_res = sum((prices[i] - predicted[i]) ** 2 for i in range(m))
    ss_tot = sum((prices[i] - mean_price) ** 2 for i in range(m))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    print(f"Precision metrics on {m} data points:")
    print(f"  MAE   (Mean Absolute Error)        : {mae:.2f}")
    print(f"  RMSE  (Root Mean Squared Error)    : {rmse:.2f}")
    print(f"  R²    (Coefficient of determination): {r2:.4f}")

    if r2 >= 0.9:
        quality = "Excellent fit"
    elif r2 >= 0.7:
        quality = "Good fit"
    elif r2 >= 0.5:
        quality = "Moderate fit"
    else:
        quality = "Poor fit"
    print(f"  => {quality}")


if __name__ == "__main__":
    main()
