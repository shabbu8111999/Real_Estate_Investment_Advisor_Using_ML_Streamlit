# Creating a Streamlit app for real estate investment advice.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import random

from src_project.config import ARTIFACTS_DIR
from src_project.data_loader import load_raw_data
from src_project.preprocessing import (
    fill_missing_values,
    add_future_price,
    add_good_investment,
)

# ---------- Cached helpers ----------
# Use Streamlit caching to avoid reloading models and data on every interaction.

@st.cache_resource
def load_models_and_schema():
    """
    Load and return the classification pipeline, regression pipeline,
    and training feature schema (X_train) from disk.
    This is cached as a resource so models are loaded only once per session.
    """
    clf_pipeline = joblib.load(ARTIFACTS_DIR / "class_model.pkl")
    reg_pipeline = joblib.load(ARTIFACTS_DIR / "reg_model.pkl")
    X_train = joblib.load(ARTIFACTS_DIR / "X_train.pkl")
    return clf_pipeline, reg_pipeline, X_train


@st.cache_data
def load_full_data():
    """
    Load raw data, apply preprocessing steps, and return the full dataframe.
    This is cached so the data processing runs once and is reused.
    """
    df = load_raw_data()
    df = fill_missing_values(df)     # fill nulls or missing values
    df = add_future_price(df)        # compute target future price column
    df = add_good_investment(df)     # add binary good-investment label
    return df


# ---------- UI helpers ----------
# Functions below create UI widgets and plots used by the app.

def build_property_form(X_train: pd.DataFrame) -> pd.DataFrame:
    """Build the property input form from the training schema.

    - Uses st.session_state to store widget values so they persist between interactions.
    - For categorical (object) columns, show a selectbox (if categories exist),
      otherwise text input.
    - For numeric columns, show a number_input with a sensible default (median).
    - Returns a single-row DataFrame containing the user's inputs.
    """
    st.subheader("Enter Property Details")

    cols = X_train.columns
    col_left, col_right = st.columns(2)  # two-column layout for compact form
    use_left = True

    # Create widget for each column in training schema
    for col in cols:
        series = X_train[col]
        key = f"input_{col}"
        container = col_left if use_left else col_right

        if series.dtype == "O":  # categorical / object column
            options = sorted(series.dropna().unique().tolist())
            if not options:
                # If no categories available, allow free text input.
                default = st.session_state.get(key, "")
                container.text_input(col, value=default, key=key)
            else:
                # Initialize session_state with first option if not set,
                # then show selectbox with the categorical options.
                if key not in st.session_state:
                    st.session_state[key] = options[0]
                container.selectbox(col, options, key=key)
        else:
            # Numeric column: default to median of that column for convenience.
            default = float(series.median())
            val = st.session_state.get(key, default)
            container.number_input(col, value=val, key=key)

        use_left = not use_left  # alternate columns

    # Collect values from session_state into a DataFrame for prediction.
    data = {col: st.session_state.get(f"input_{col}") for col in cols}
    return pd.DataFrame([data])


def show_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build sidebar filters to narrow down properties shown in the dataset.
    - Adds price range slider, BHK multiselect, and area range slider if columns exist.
    - Returns a filtered dataframe based on the sidebar controls.
    """
    st.sidebar.header("Filter Properties")
    filtered_df = df.copy()

    # Try to find a price column (case-insensitive)
    price_cols = [c for c in df.columns if "Price" in c or "price" in c]
    if price_cols:
        price_col = price_cols[0]
        min_p, max_p = float(df[price_col].min()), float(df[price_col].max())
        p_min, p_max = st.sidebar.slider(
            f"{price_col} range", min_value=min_p, max_value=max_p, value=(min_p, max_p)
        )
        filtered_df = filtered_df[
            (filtered_df[price_col] >= p_min) & (filtered_df[price_col] <= p_max)
        ]

    # BHK filter (if column available)
    if "BHK" in df.columns:
        bhk_values = sorted(df["BHK"].dropna().unique().tolist())
        selected_bhk = st.sidebar.multiselect("BHK", bhk_values, default=bhk_values)
        if selected_bhk:
            filtered_df = filtered_df[filtered_df["BHK"].isin(selected_bhk)]

    # Area column detection (check common names)
    area_col = None
    for c in ["Area_in_SqFt", "Area", "total_sqft", "Total_Sqft"]:
        if c in df.columns:
            area_col = c
            break

    if area_col is not None:
        min_a, max_a = float(df[area_col].min()), float(df[area_col].max())
        a_min, a_max = st.sidebar.slider(
            f"{area_col} range", min_value=min_a, max_value=max_a, value=(min_a, max_a)
        )
        filtered_df = filtered_df[
            (filtered_df[area_col] >= a_min) & (filtered_df[area_col] <= a_max)
        ]

    st.sidebar.write(f"Showing {len(filtered_df)} properties after filters.")
    return filtered_df


def plot_location_insights(df: pd.DataFrame):
    """
    Plot basic location-wise insights:
    - Top locations by median price per sqft (if that column exists)
    - Good investment rate by location (if 'Good_Investment' exists)
    """
    st.subheader("Location-wise Insights")

    # Decide which column stores location information
    location_col = "City" if "City" in df.columns else "Location" if "Location" in df.columns else None
    if location_col is None:
        st.info("No 'City' or 'Location' column found.")
        return

    # Plot median price per sqft by location (top 10)
    ppsf_col = None
    for c in ["Price_per_SqFt", "price_per_sqft", "Price_per_sqft"]:
        if c in df.columns:
            ppsf_col = c
            break

    if ppsf_col is not None:
        group_ppsf = (
            df.groupby(location_col)[ppsf_col]
            .median()
            .sort_values(ascending=False)
            .head(10)
        )
        fig, ax = plt.subplots(figsize=(4, 3))
        group_ppsf.plot(kind="bar", ax=ax)
        ax.set_ylabel(ppsf_col)
        ax.set_title("Top locations by median price/sqft")
        st.pyplot(fig)

    # Plot rate of 'Good_Investment' by location (top 10)
    if "Good_Investment" in df.columns:
        good_rate = (
            df.groupby(location_col)["Good_Investment"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        fig, ax = plt.subplots(figsize=(4, 3))
        good_rate.plot(kind="bar", ax=ax)
        ax.set_ylabel("Good investment rate")
        ax.set_title("Good Investment rate by location")
        st.pyplot(fig)


def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Show a simple correlation heatmap for numeric columns.
    - Uses matplotlib's imshow for a compact display.
    - If there are no numeric columns, show an info message.
    """
    st.subheader("Numeric Correlation Heatmap")

    num_df = df.select_dtypes(include=["int64", "float64"])
    if num_df.empty:
        st.info("No numeric columns for correlation.")
        return

    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(corr, cmap="viridis")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.columns, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    st.pyplot(fig)


def plot_feature_importance(clf_pipeline):
    """
    Plot top feature importances from the classification pipeline (if available).
    - Attempts to extract preprocessor feature names and model importances.
    - If something fails (different pipeline shapes), shows an info message.
    """
    st.subheader("Top Features")

    try:
        preproc = clf_pipeline.named_steps["preprocessor"]
        model = clf_pipeline.named_steps["model"]
        feature_names = preproc.get_feature_names_out()
        importances = model.feature_importances_
    except Exception as e:
        st.info(f"Could not compute feature importances: {e}")
        return

    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(10)
        .set_index("feature")
    )

    fig, ax = plt.subplots(figsize=(4, 3))
    fi_df.plot(kind="bar", ax=ax, legend=False)
    ax.set_ylabel("Importance")
    st.pyplot(fig)


def compute_market_score(input_df: pd.DataFrame, market_df: pd.DataFrame) -> float:
    """
    Compute a simple market score (0-100) for the input property.
    - Starts at 50 and adjusts based on price per sqft vs city median,
      BHK, area per BHK, and property age.
    - Uses safe checks so missing or invalid values don't break the logic.
    """
    row = input_df.iloc[0]
    score = 50.0

    # Compare Price_per_SqFt to city median (if available)
    if (
        "Price_per_SqFt" in market_df.columns
        and "Price_per_SqFt" in input_df.columns
        and "City" in market_df.columns
        and "City" in input_df.columns
    ):
        city = row["City"]
        ppsf = float(row["Price_per_SqFt"])
        city_df = market_df[market_df["City"] == city]

        # Fallback to market-wide median if no city data
        if not city_df.empty:
            city_median = city_df["Price_per_SqFt"].median()
        else:
            city_median = market_df["Price_per_SqFt"].median()

        # Only adjust score if medians are finite and positive
        if np.isfinite(city_median) and city_median > 0 and np.isfinite(ppsf):
            diff_pct = (city_median - ppsf) / city_median
            # Clamp difference to avoid extreme swings
            diff_pct = max(-0.5, min(0.5, diff_pct))
            score += diff_pct * 60  # scale effect

    # Small bonus/penalty for common BHK values
    if "BHK" in input_df.columns:
        try:
            bhk = float(row["BHK"])
            if 1 <= bhk <= 3:
                score += 5
            elif bhk >= 5:
                score -= 5
        except Exception:
            # ignore invalid BHK values
            pass

    # Prefer comfortable area per BHK (simple heuristic)
    area_col = None
    for c in ["Area_in_SqFt", "Area", "total_sqft", "Total_Sqft"]:
        if c in input_df.columns:
            area_col = c
            break

    if area_col and "BHK" in input_df.columns:
        try:
            area = float(row[area_col])
            bhk = float(row["BHK"])
            if bhk > 0:
                area_per_bhk = area / bhk
                if 450 <= area_per_bhk <= 900:
                    score += 5
                elif area_per_bhk < 350 or area_per_bhk > 1100:
                    score -= 3
        except Exception:
            pass

    # Age penalty/bonus: newer properties slightly preferred
    age_col = None
    for c in ["Property_Age", "Age", "age"]:
        if c in input_df.columns:
            age_col = c
            break

    if age_col:
        try:
            age = float(row[age_col])
            if age <= 10:
                score += 5
            elif age > 25:
                score -= 5
        except Exception:
            pass

    # Clamp final score to 0..100
    return max(0, min(100, score))


def get_unique_good_input(full_df, used_indices):
    """
    Return a random 'Good_Investment' example that hasn't been used yet.
    - Keeps track of used indices to avoid repeating examples in the session.
    - Returns None when all examples have been used.
    """
    df = full_df.copy()
    df = df[df["Good_Investment"] == 1].reset_index(drop=True)

    available_indices = [i for i in range(len(df)) if i not in used_indices]
    if not available_indices:
        return None

    idx = random.choice(available_indices)
    used_indices.add(idx)
    return df.iloc[idx]


# ---------- Main app ----------
def main():
    # Page settings
    st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

    st.title("🏠 Real Estate Investment Advisor")
    st.write("Use the filters and form below to analyze properties and check if they are a good investment.")

    # Load models and full dataframe (cached)
    clf_pipeline, reg_pipeline, X_train = load_models_and_schema()
    full_df = load_full_data()

    # Sidebar filters applied to the full dataset
    filtered_df = show_filters(full_df)

    # Tabs for different sections of the app
    tab1, tab2, tab3 = st.tabs(
        ["🔍 Investment Prediction", "📊 Market Insights", "🧠 Model Explainability"]
    )

    # ----- Tab 1: Prediction -----
    with tab1:
        st.subheader("Property Input Form")

        # Initialize used indices set for auto-fill examples (persist across interactions)
        if "used_good_indices" not in st.session_state:
            st.session_state.used_good_indices = set()

        # Auto-fill button populates the form with a real good-investment example
        # (uses get_unique_good_input to avoid repeats)
        if st.button("✨ Auto-Fill Good Example"):
            row = get_unique_good_input(full_df, st.session_state.used_good_indices)
            if row is not None:
                for col in X_train.columns:
                    st.session_state[f"input_{col}"] = row[col]
                st.success("Auto-filled with a real good-investment example!")
            else:
                st.warning("No more new sample properties left!")

        # Build the form (reads/writes st.session_state)
        input_df = build_property_form(X_train)

        # When user clicks Predict: run both classification and regression models
        if st.button("Predict"):
            class_pred = clf_pipeline.predict(input_df)[0]
            proba = clf_pipeline.predict_proba(input_df)[0][1]  # probability for class 1
            future_price = reg_pipeline.predict(input_df)[0]
            market_df = filtered_df if not filtered_df.empty else full_df

            st.markdown("---")
            st.subheader("Prediction Results")

            # Show readable result text and short explanation
            if class_pred == 1:
                st.success("🎯 Good Investment")
                st.caption("This property appears **reasonably priced or undervalued** compared to the market.")
            else:
                st.error("❌ Not Recommended")
                st.caption("This property seems **overpriced vs similar properties** in the market.")

            # Show estimated future price and a confidence progress bar
            st.metric("Estimated Price After 5 Years", f"₹ {future_price:,.2f}")
            st.progress(float(proba))
            st.caption(f"Confidence: {proba:.2f}")

            # Small market comparison chart: compare your property's ppsf to market median
            st.markdown("---")
            st.subheader("📉 Price Position")

            if "Price_per_SqFt" in input_df.columns and "Price_per_SqFt" in market_df.columns:
                ppsf = float(input_df["Price_per_SqFt"].iloc[0])
                median_ppsf = float(market_df["Price_per_SqFt"].median())

                fig, ax = plt.subplots(figsize=(4, 3))  # small chart
                bars = ax.bar(["You", "Market"], [ppsf, median_ppsf])
                # color bar green when recommended, red when not
                bars[0].set_color("green" if class_pred == 1 else "red")
                ax.set_ylim(min(ppsf, median_ppsf) * 0.8,
                            max(ppsf, median_ppsf) * 1.2)
                ax.set_ylabel("₹ / SqFt")
                st.pyplot(fig)
                st.caption("If **Your** bar is below **Market**, the deal is generally better.")

            # Compute and display a simple market score (0-100)
            score = compute_market_score(input_df, market_df)
            st.metric("Market Score", f"{score:.0f}/100")
            st.progress(score / 100)


    # ----- Tab 2: Data & Charts -----
    with tab2:
        st.subheader("Filtered Properties")
        # Show first 200 rows of the filtered dataframe in a scrollable table
        st.dataframe(filtered_df.head(200))

        st.markdown("---")
        plot_location_insights(filtered_df)

        st.markdown("---")
        plot_correlation_heatmap(filtered_df)

    # ----- Tab 3: Explainability -----
    with tab3:
        # Show top features from the trained classifier
        plot_feature_importance(clf_pipeline)


if __name__ == "__main__":
    main()
