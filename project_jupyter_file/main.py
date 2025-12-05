
import streamlit as st
import pandas as pd
import joblib
import os

# Paths were the models is saved

ARTIFACTS_DIR = "project_jupyter_file/artifacts"
CLASS_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "class_model.pkl")
REG_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "reg_model.pkl")


@st.cache_resource
def load_models():
    """
    Load the trained classification and regression models
    from the artifacts folder.
    This runs only once and then caches the models.
    """
    if not os.path.exists(CLASS_MODEL_PATH):
        raise FileNotFoundError(f"Classification model not found at {CLASS_MODEL_PATH}")
    if not os.path.exists(REG_MODEL_PATH):
        raise FileNotFoundError(f"Regression model not found at {REG_MODEL_PATH}")

    class_model = joblib.load(CLASS_MODEL_PATH)
    reg_model = joblib.load(REG_MODEL_PATH)
    return class_model, reg_model


# Trying to load models
try:
    class_model, reg_model = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    error_msg = str(e)


# This is Streamlit UI Layout

st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

st.title("🏠 Real Estate Investment Advisor")

st.write(
    """
This app uses Machine Learning models to:

1. Predict whether a property is a **Good Investment** or not  
2. Estimate the **Future Price after 5 years**  

The models are trained on Indian housing data.
"""
)

# Show model status
if models_loaded:
    st.success("✅ Models loaded successfully from 'artifacts' folder.")
else:
    st.error("❌ Failed to load models.")
    st.code(error_msg)
    st.stop()  # Stop the app here if models not loaded


st.sidebar.header("ℹ About this App")
st.sidebar.write("**Task 1:** Classification → Good Investment (Yes/No)")
st.sidebar.write("**Task 2:** Regression → Future Price after 5 years (in Lakhs)")
st.sidebar.write("Models are trained in a separate notebook/file and saved as `.pkl`.")


# Input form for Property

st.header("Enter Property Details")

# We use the same feature names that were used for training:
# ['State', 'City', 'Locality', 'Property_Type', 'BHK', 'Size_in_SqFt',
# 'Price_in_Lakhs', 'Year_Built', 'Furnished_Status', 'Parking_Space',
# 'Facing', 'Nearby_Schools', 'Nearby_Hospitals',
# 'Public_Transport_Accessibility', 'Amenities',
# 'Floor_No', 'Total_Floors', 'Security', 'Availability_Status',
# 'Owner_Type', 'Age_of_Property', 'Price_per_SqFt']


col1, col2 = st.columns(2)

with col1:
    state = st.text_input("State", "Maharashtra")
    city = st.text_input("City", "Mumbai")
    locality = st.text_input("Locality", "Andheri East")

    property_type = st.selectbox(
        "Property Type",
        ["Apartment", "Villa", "Independent House", "Other"]
    )

    bhk = st.number_input("BHK", min_value=1, max_value=10, value=2, step=1)
    size_in_sqft = st.number_input(
        "Size (SqFt)", min_value=100.0, max_value=20000.0, value=1000.0, step=50.0
    )
    price_in_lakhs = st.number_input(
        "Current Price (Lakhs)", min_value=5.0, max_value=10000.0, value=100.0, step=1.0
    )

with col2:
    year_built = st.number_input(
        "Year Built", min_value=1950, max_value=2025, value=2015, step=1
    )

    furnished_status = st.selectbox(
        "Furnished Status",
        ["Unfurnished", "Semi-Furnished", "Fully-Furnished"]
    )

    parking_space = st.selectbox(
        "Parking Space",
        ["None", "1", "2", "3+"]
    )

    facing = st.selectbox(
        "Facing Direction",
        ["East", "West", "North", "South",
         "North-East", "North-West", "South-East", "South-West"]
    )

    nearby_schools = st.number_input(
        "Nearby Schools (within 5 km)", min_value=0, max_value=50, value=3, step=1
    )

    nearby_hospitals = st.number_input(
        "Nearby Hospitals (within 5 km)", min_value=0, max_value=50, value=2, step=1
    )

    public_transport = st.selectbox(
        "Public Transport Accessibility",
        ["Low", "Medium", "High"]
    )

    amenities = st.selectbox(
        "Amenities (Gym/Pool/Club etc.)",
        ["Yes", "No"]
    )

    floor_no = st.number_input(
        "Floor Number (0 if Ground)", min_value=0, max_value=200, value=1, step=1
    )

    total_floors = st.number_input(
        "Total Floors in Building", min_value=1, max_value=200, value=10, step=1
    )

    security = st.selectbox(
        "Security Type",
        ["None", "Gated", "CCTV", "Gated + CCTV"]
    )

    availability_status = st.selectbox(
        "Availability Status",
        ["Ready to Move", "Under Construction", "Sold"]
    )

    owner_type = st.selectbox(
        "Owner Type",
        ["Owner", "Builder", "Agent", "Other"]
    )



# Prediction Buttons for API

CURRENT_YEAR = 2025
age_of_property = CURRENT_YEAR - year_built

# Same formula you used during training:
price_per_sqft = (price_in_lakhs * 100000) / size_in_sqft

if st.button("Predict Investment & Future Price"):
    # Build a single-row DataFrame with same columns as during training
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
    st.write(input_df)

   # Calling the Models

    # Classification: Good Investment (0/1)
    class_pred = class_model.predict(input_df)[0]

    # Probability (confidence) if available
    if hasattr(class_model, "predict_proba"):
        class_proba = class_model.predict_proba(input_df)[0][1]  # probability of class 1
    else:
        class_proba = None

    # Regression: Future Price after 5 years
    future_price_pred = reg_model.predict(input_df)[0]


    # 6. Showing the Results

    st.subheader("Prediction Results")

    if class_pred == 1:
        st.success("✅ This property is predicted as a **Good Investment**.")
    else:
        st.warning("⚠ This property is predicted as **Not a strong investment**.")

    if class_proba is not None:
        st.write(f"Confidence (Good Investment probability): **{class_proba:.2f}**")

    st.write(f"💰 Estimated Future Price after 5 years: **₹ {future_price_pred:.2f} Lakhs**")

    st.info(
        "Note: These predictions are based on historical data and models. "
        "They should be used as guidance, not as guaranteed financial advice."
    )
