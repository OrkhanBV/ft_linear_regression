"""
Bonus: Plot data points and the regression line on the same graph.
Requires matplotlib (pip install matplotlib).
"""
import csv
import json
import os
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

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
        print("Note: no model file found. Only raw data will be plotted.")
        return None, None
    try:
        with open(MODEL_FILE) as f:
            data = json.load(f)
        return float(data["theta0"]), float(data["theta1"])
    except Exception:
        print("Warning: could not load model. Only raw data will be plotted.")
        return None, None


def main():
    km, prices = load_data(DATA_FILE)
    theta0, theta1 = load_thetas()

    plt.figure(figsize=(10, 6))
    plt.scatter(km, prices, color="steelblue", zorder=5, label="Data points")

    if theta0 is not None and theta1 is not None:
        x_min, x_max = min(km), max(km)
        x_line = [x_min, x_max]
        y_line = [theta0 + theta1 * x for x in x_line]
        plt.plot(
            x_line, y_line,
            color="crimson", linewidth=2,
            label=f"Regression line\nθ0={theta0:.2f}, θ1={theta1:.6f}"
        )

    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (€)")
    plt.title("Car Price vs Mileage — ft_linear_regression")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
