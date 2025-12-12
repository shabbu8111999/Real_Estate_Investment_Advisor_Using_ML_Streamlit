from src_project.data_loader import load_raw_data
from src_project.preprocessing import (
    fill_missing_values,
    add_future_price,
    add_good_investment,
    make_features_targets,
)

def main():
    # Load + preprocess exactly like training
    df = load_raw_data()
    df = fill_missing_values(df)
    df = add_future_price(df)
    df = add_good_investment(df)

    # Build features/targets like in training
    X, y_class, y_reg = make_features_targets(df)

    # Keep only rows where Good_Investment == 1
    good_mask = df["Good_Investment"] == 1
    X_good = X[good_mask]

    print(f"Total rows with Good_Investment = 1: {len(X_good)}")

    # Take sample “perfect” inputs
    sample = X_good.sample(min(15, len(X_good)))  # no random_state

    # Print out all examples
    for i in range(len(sample)):
        print(f"\n--- Example #{i+1} ---")
        row = sample.iloc[i]
        for col, val in row.items():
            print(f"{col}: {val}")

    print("\n=== One good-investment input (copy this into Streamlit) ===")
    for col, val in sample.iloc[0].items():
        print(f"{col}: {val}")

    print("\n=== Same row as dict (easy copy) ===")
    print(sample.iloc[0].to_dict())


if __name__ == "__main__":
    main()
