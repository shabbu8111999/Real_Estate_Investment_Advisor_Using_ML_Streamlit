
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

ARTIFACTS_DIR = "project_jupyter_file/artifacts"
CLASS_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "class_model.pkl")
REG_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "reg_model.pkl")
DATA_PATH = "project_jupyter_file/india_housing_prices.csv"

st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

@st.cache_resource
def load_models():
    if not os.path.exists(CLASS_MODEL_PATH):
        raise FileNotFoundError(f"Classification model not found at {CLASS_MODEL_PATH}")
    if not os.path.exists(REG_MODEL_PATH):
        raise FileNotFoundError(f"Regression model not found at {REG_MODEL_PATH}")
    return joblib.load(CLASS_MODEL_PATH), joblib.load(REG_MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if "Price_per_SqFt" not in df.columns:
        df["Price_per_SqFt"] = (df["Price_in_Lakhs"] * 100000) / df["Size_in_SqFt"]
    if "Age_of_Property" not in df.columns and "Year_Built" in df.columns:
        df["Age_of_Property"] = 2025 - df["Year_Built"]
    return df

def get_feature_importance(pipe):
    try:
        pre = pipe.named_steps["preprocessor"]
        model = pipe.named_steps["model"]
        feats = pre.get_feature_names_out()
        imps = model.feature_importances_
        return pd.DataFrame({"feature": feats, "importance": imps}).sort_values(
            "importance", ascending=False
        )
    except Exception:
        return None

# load
try:
    class_model, reg_model = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    model_error = str(e)

try:
    df_data = load_data()
    data_loaded = True
except Exception as e:
    data_loaded = False
    data_error = str(e)

fi_df = get_feature_importance(class_model) if models_loaded else None

# header
st.title("🏠 Real Estate Investment Advisor")
st.write(
    """
Predict:
1. **Good Investment or Not** (Classification)  
2. **Estimated Price after 5 Years** (Regression)
"""
)

c1, c2 = st.columns(2)
with c1:
    if models_loaded:
        st.success("Models loaded.")
    else:
        st.error("Model load failed.")
        st.code(model_error)
        st.stop()
with c2:
    if data_loaded:
        st.info("Dataset loaded for filters & insights.")
    else:
        st.warning("Dataset not available for insights.")
        st.code(data_error)

st.sidebar.header("About")
st.sidebar.write("- Good Investment classification")
st.sidebar.write("- Future price prediction")
st.sidebar.write("- Filters + charts for insights")

tab_pred, tab_insights = st.tabs(["🔮 Prediction", "📊 Insights & Filters"])

# PREDICTION TAB
with tab_pred:
    st.header("Property Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        state = st.text_input("State", "Maharashtra")
        city = st.text_input("City", "Mumbai")
        locality = st.text_input("Locality", "Andheri East")
        property_type = st.selectbox("Property Type", ["Apartment", "Villa", "Independent House", "Other"])
        bhk = st.number_input("BHK", 1, 10, 2, 1)
        floor_no = st.number_input("Floor Number (0 = Ground)", 0, 200, 1, 1)
        total_floors = st.number_input("Total Floors", 1, 200, 10, 1)

    with col2:
        size_in_sqft = st.number_input("Size (SqFt)", 100.0, 20000.0, 1000.0, 50.0)
        price_in_lakhs = st.number_input("Current Price (Lakhs)", 5.0, 10000.0, 100.0, 1.0)
        year_built = st.number_input("Year Built", 1950, 2025, 2015, 1)
        parking_space = st.selectbox("Parking Space", ["None", "1", "2", "3+"])
        facing = st.selectbox(
            "Facing Direction",
            ["East", "West", "North", "South", "North-East", "North-West", "South-East", "South-West"],
        )

    with col3:
        furnished_status = st.selectbox("Furnished Status", ["Unfurnished", "Semi-Furnished", "Fully-Furnished"])
        amenities = st.selectbox("Amenities (Gym/Pool/Club)", ["Yes", "No"])
        nearby_schools = st.number_input("Nearby Schools (5 km)", 0, 50, 3, 1)
        nearby_hospitals = st.number_input("Nearby Hospitals (5 km)", 0, 50, 2, 1)
        public_transport = st.selectbox("Public Transport Access", ["Low", "Medium", "High"])
        security = st.selectbox("Security Type", ["None", "Gated", "CCTV", "Gated + CCTV"])
        availability_status = st.selectbox("Availability Status", ["Ready to Move", "Under Construction", "Sold"])
        owner_type = st.selectbox("Owner Type", ["Owner", "Builder", "Agent", "Other"])

    age_of_property = 2025 - year_built
    price_per_sqft = (price_in_lakhs * 100000) / size_in_sqft if size_in_sqft > 0 else 0

    predict_btn = st.button("🚀 Predict Investment & Future Price", use_container_width=True)

    if predict_btn:
        input_data = {
            "State": [state],
            "City": [city],
            "Locality": [locality],
            "Property_Type": [property_type],
            "BHK": [bhk],
            "Size_in_SqFt": [size_in_sqft],
            "Price_in_Lakhs": [price_in_lakhs],
            "Year_Built": [year_built],
            "Furnished_Status": [furnished_status],
            "Parking_Space": [parking_space],
            "Facing": [facing],
            "Nearby_Schools": [nearby_schools],
            "Nearby_Hospitals": [nearby_hospitals],
            "Public_Transport_Accessibility": [public_transport],
            "Amenities": [amenities],
            "Floor_No": [floor_no],
            "Total_Floors": [total_floors],
            "Security": [security],
            "Availability_Status": [availability_status],
            "Owner_Type": [owner_type],
            "Age_of_Property": [age_of_property],
            "Price_per_SqFt": [price_per_sqft],
        }

        input_df = pd.DataFrame(input_data)

        st.subheader("Input Summary")
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

        class_pred = class_model.predict(input_df)[0]
        class_proba = class_model.predict_proba(input_df)[0][1] if hasattr(class_model, "predict_proba") else None
        future_price = reg_model.predict(input_df)[0]

        c_res1, c_res2 = st.columns(2)
        with c_res1:
            if class_pred == 1:
                st.success("✅ Good Investment")
            else:
                st.warning("⚠ Not a Strong Investment")
            if class_proba is not None:
                st.write(f"Model confidence: **{class_proba:.2f}**")

        with c_res2:
            st.metric("Estimated Price after 5 Years", f"₹ {future_price:.2f} Lakhs")

        st.info("Predictions are based on historical data and are for guidance only.")

        st.subheader("Feature Importance (Classification Model)")
        if fi_df is not None:
            top_fi = fi_df.head(12)
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=top_fi, x="importance", y="feature", ax=ax)
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            st.pyplot(fig)
        else:
            st.caption("Feature importance not available for this model.")

# INSIGHTS TAB 
with tab_insights:
    st.header("Filters & Market Insights")

    if not data_loaded:
        st.error("Dataset not available.")
    else:
        f1, f2, f3 = st.columns(3)
        with f1:
            cities = ["All"] + sorted(df_data["City"].dropna().unique().tolist())
            filt_city = st.selectbox("City", cities)
        with f2:
            pmin, pmax = float(df_data["Price_in_Lakhs"].min()), float(df_data["Price_in_Lakhs"].max())
            filt_price = st.slider("Price Range (Lakhs)", pmin, pmax, (pmin, pmax))
        with f3:
            bhk_vals = sorted(df_data["BHK"].dropna().unique().tolist())
            filt_bhk = st.multiselect("BHK", bhk_vals, default=bhk_vals)

        df_f = df_data.copy()
        if filt_city != "All":
            df_f = df_f[df_f["City"] == filt_city]
        df_f = df_f[(df_f["Price_in_Lakhs"] >= filt_price[0]) & (df_f["Price_in_Lakhs"] <= filt_price[1])]
        if filt_bhk:
            df_f = df_f[df_f["BHK"].isin(filt_bhk)]

        st.write(f"Showing **{len(df_f)}** properties (filtered).")
        st.dataframe(df_f.head(50), use_container_width=True)

        st.markdown("---")
        c_a, c_b = st.columns(2)

        with c_a:
            st.subheader("Price Distribution (Filtered)")
            fig1, ax1 = plt.subplots(figsize=(5, 3))
            sns.histplot(df_f["Price_in_Lakhs"], kde=True, ax=ax1)
            ax1.set_xlabel("Price (Lakhs)")
            st.pyplot(fig1)

        with c_b:
            st.subheader("BHK Distribution (Filtered)")
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            sns.countplot(data=df_f, x="BHK", ax=ax2)
            ax2.set_xlabel("BHK")
            st.pyplot(fig2)

        st.markdown("---")
        st.subheader("City vs BHK – Avg Price per SqFt (Heatmap)")
        if not df_f.empty:
            hm = df_f.pivot_table(index="City", columns="BHK", values="Price_per_SqFt", aggfunc="mean")
            fig3, ax3 = plt.subplots(figsize=(7, 4))
            sns.heatmap(hm, cmap="YlOrRd", ax=ax3)
            ax3.set_xlabel("BHK")
            ax3.set_ylabel("City")
            st.pyplot(fig3)
        else:
            st.caption("No data for selected filters.")

        st.markdown("---")
        st.subheader("Year Built vs Median Price (Trend)")
        if "Year_Built" in df_f.columns and not df_f.empty:
            trend = df_f.groupby("Year_Built")["Price_in_Lakhs"].median().reset_index().sort_values("Year_Built")
            fig4, ax4 = plt.subplots(figsize=(7, 3))
            ax4.plot(trend["Year_Built"], trend["Price_in_Lakhs"], marker="o")
            ax4.set_xlabel("Year Built")
            ax4.set_ylabel("Median Price (Lakhs)")
            st.pyplot(fig4)
        else:
            st.caption("Trend not available for selected filters.")
