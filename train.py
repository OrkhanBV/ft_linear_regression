import csv
import json
import os
import sys

DATA_FILE = "data.csv"
MODEL_FILE = "model.json"

LEARNING_RATE = 0.1
ITERATIONS = 1000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(filepath):
    """
    Parse a CSV file with 'km' and 'price' columns.
    Skips rows with invalid or negative values with a warning.
    Exits on missing file or missing columns.
    """
    mileages = []
    prices = []

    try:
        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            if headers is None or "km" not in headers or "price" not in headers:
                print(f"Error: CSV must have 'km' and 'price' columns. Found: {headers}")
                sys.exit(1)
            for i, row in enumerate(reader, start=2):
                try:
                    km = float(row["km"])
                    price = float(row["price"])
                except (ValueError, KeyError):
                    print(f"Warning: skipping row {i} (invalid values): {row}")
                    continue
                if km < 0 or price < 0:
                    print(f"Warning: skipping row {i} (negative value): {row}")
                    continue
                mileages.append(km)
                prices.append(price)

    except FileNotFoundError:
        print(f"Error: dataset file '{filepath}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        sys.exit(1)

    if len(mileages) < 2:
        print("Error: need at least 2 valid data points to train.")
        sys.exit(1)

    return mileages, prices


# ---------------------------------------------------------------------------
# Feature normalization (z-score)
# ---------------------------------------------------------------------------

def compute_mean(values):
    return sum(values) / len(values)


def compute_std(values, mean_val):
    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
    return variance ** 0.5


def normalize(values):
    """
    Z-score normalize a list of values.
    Returns (normalized_list, mean, std).
    Raises ValueError if std == 0 (all values identical).
    """
    mean_val = compute_mean(values)
    std_val = compute_std(values, mean_val)
    if std_val == 0:
        raise ValueError("All mileage values are identical — cannot normalize (std = 0).")
    return [(x - mean_val) / std_val for x in values], mean_val, std_val


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------

def estimate_price(mileage, theta0, theta1):
    """Linear hypothesis: estimatePrice(mileage) = theta0 + (theta1 * mileage)"""
    return theta0 + (theta1 * mileage)


# ---------------------------------------------------------------------------
# Gradient descent (on normalized mileage)
# ---------------------------------------------------------------------------

def gradient_descent(km_norm, prices, learning_rate, iterations):
    """
    Runs gradient descent using the exact formulas from the subject.

    tmpTheta0 = learningRate * (1/m) * sum(estimatePrice(km[i]) - price[i])
    tmpTheta1 = learningRate * (1/m) * sum((estimatePrice(km[i]) - price[i]) * km[i])

    Both thetas are updated simultaneously (tmp values computed before any update).
    """
    theta0 = 0.0
    theta1 = 0.0
    m = len(km_norm)

    for _ in range(iterations):
        # Compute gradients using current theta0 and theta1 (simultaneous update)
        sum0 = 0.0
        sum1 = 0.0
        for i in range(m):
            error = estimate_price(km_norm[i], theta0, theta1) - prices[i]
            sum0 += error
            sum1 += error * km_norm[i]

        tmp_theta0 = learning_rate * (1.0 / m) * sum0
        tmp_theta1 = learning_rate * (1.0 / m) * sum1

        # Simultaneous update
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

    return theta0, theta1


# ---------------------------------------------------------------------------
# Denormalization: convert thetas back to real-scale values
# ---------------------------------------------------------------------------

def denormalize_thetas(theta0_norm, theta1_norm, mean_km, std_km):
    """
    Training is done on normalized km: km_norm = (km - mean_km) / std_km

    The model during training is:
        price = theta0_norm + theta1_norm * km_norm
              = theta0_norm + theta1_norm * (km - mean_km) / std_km

    Rearranging to the form theta0_real + theta1_real * km:
        theta1_real = theta1_norm / std_km
        theta0_real = theta0_norm - theta1_norm * mean_km / std_km

    This lets predict.py use the real formula without knowing about normalization.
    """
    theta1_real = theta1_norm / std_km
    theta0_real = theta0_norm - theta1_norm * mean_km / std_km
    return theta0_real, theta1_real


# ---------------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------------

def save_model(theta0, theta1):
    try:
        with open(MODEL_FILE, "w") as f:
            json.dump({"theta0": theta0, "theta1": theta1}, f, indent=2)
    except IOError as e:
        print(f"Error: could not write model file: {e}")
        sys.exit(1)
    print(f"Training complete.")
    print(f"  theta0 = {theta0:.6f}")
    print(f"  theta1 = {theta1:.10f}")
    print(f"  Model saved to '{MODEL_FILE}'.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mileages, prices = load_data(DATA_FILE)
    print(f"Loaded {len(mileages)} data points from '{DATA_FILE}'.")

    try:
        km_norm, mean_km, std_km = normalize(mileages)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Training with learning_rate={LEARNING_RATE}, iterations={ITERATIONS}...")
    theta0_norm, theta1_norm = gradient_descent(
        km_norm, prices, LEARNING_RATE, ITERATIONS
    )

    theta0_real, theta1_real = denormalize_thetas(
        theta0_norm, theta1_norm, mean_km, std_km
    )
    save_model(theta0_real, theta1_real)


if __name__ == "__main__":
    main()
