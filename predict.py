import json
import os
import sys

MODEL_FILE = "model.json"


def load_thetas():
    """
    Load theta0 and theta1 from model.json.
    Returns (0.0, 0.0) if the file is missing or corrupted — as required by the subject.
    """
    if not os.path.exists(MODEL_FILE):
        print("Note: no trained model found. Using theta0=0 and theta1=0.")
        return 0.0, 0.0
    try:
        with open(MODEL_FILE, "r") as f:
            data = json.load(f)
        theta0 = float(data["theta0"])
        theta1 = float(data["theta1"])
        return theta0, theta1
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        print("Warning: model file is corrupted. Using theta0=0 and theta1=0.")
        return 0.0, 0.0


def estimate_price(mileage, theta0, theta1):
    """Linear hypothesis: estimatePrice(mileage) = theta0 + (theta1 * mileage)"""
    return theta0 + (theta1 * mileage)


def get_mileage():
    """Prompt user for a mileage value. Loops until valid input is given."""
    while True:
        try:
            raw = input("Enter mileage (km): ").strip()
            if not raw:
                print("Error: input cannot be empty.")
                continue
            mileage = float(raw)
            if mileage < 0:
                print("Error: mileage cannot be negative.")
                continue
            return mileage
        except ValueError:
            print("Error: please enter a valid number.")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)


def main():
    theta0, theta1 = load_thetas()
    mileage = get_mileage()
    price = estimate_price(mileage, theta0, theta1)
    if price < 0:
        print(f"Estimated price: {price:.2f} (warning: model predicts negative price at this mileage)")
    else:
        print(f"Estimated price: {price:.2f}")


if __name__ == "__main__":
    main()
