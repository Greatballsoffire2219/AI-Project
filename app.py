
import json
import joblib
import pandas as pd
import streamlit as st

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="EstateIQ | Property Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# -------------------------------
# LOAD MODEL AND LOCALITIES
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.joblib")

@st.cache_data
def load_locations():
    with open("locations.json", "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
locations = load_locations()

# -------------------------------
# AMENITY FEATURES
# -------------------------------
amenities = [
    "Gymnasium",
    "Lift Available",
    "Car Parking",
    "Maintenance Staff",
    "24x7 Security",
    "Children",
    "Clubhouse",
    "Intercom",
    "Landscaped Gardens",
    "Indoor Games",
    "Gas Connection",
    "Jogging Track",
    "Swimming Pool"
]

# -------------------------------
# HEADER
# -------------------------------
st.title("🏡 EstateIQ")
st.subheader("AI-Powered House Price Prediction")
st.write(
    "Estimate a property's price using its location, "
    "size, bedrooms and available amenities."
)

st.divider()

# -------------------------------
# PROPERTY DETAILS
# -------------------------------
st.header("1. Property Details")

with st.form("property_form"):
    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            "Built-up area (sq ft)",
            min_value=100,
            max_value=20000,
            value=1050,
            step=50
        )

        bedrooms = st.selectbox(
            "Number of bedrooms",
            options=[1, 2, 3, 4, 5, 6, 7, 8]
        )

    with col2:
        location = st.selectbox(
            "Property location",
            options=locations
        )

        resale_code = st.selectbox(
            "New/Resale dataset code",
            options=[0, 1],
            help=(
                "Select the original dataset's numeric code. "
                "Confirm which code means New and which means "
                "Resale before assigning human-readable labels."
            )
        )

    st.divider()
    st.header("2. Amenities")
    st.caption("Select the amenities available at this property.")

    amenity_values = {}

    amenity_columns = st.columns(3)

    for i, amenity in enumerate(amenities):
        with amenity_columns[i % 3]:
            amenity_values[amenity] = st.checkbox(
                amenity,
                value=True if amenity in [
                    "Lift Available",
                    "Car Parking",
                    "24x7 Security"
                ] else False
            )

    st.divider()

    submitted = st.form_submit_button(
        "Estimate Property Price",
        type="primary",
        use_container_width=True
    )

# -------------------------------
# PREDICTION
# -------------------------------
if submitted:
    input_data = {
        "Area": area,
        "No. of Bedrooms": bedrooms,
        "New/Resale": resale_code,
        **{
            name: int(value)
            for name, value in amenity_values.items()
        },
        "Location": location
    }

    input_df = pd.DataFrame([input_data])

    try:
        predicted_price = float(model.predict(input_df)[0])

        if predicted_price <= 0:
            st.error(
                "The model returned a non-positive price. "
                "Try another property configuration."
            )
        else:
            st.divider()
            st.header("3. Estimated Property Price")

            lakh_price = predicted_price / 100000

            if lakh_price >= 100:
                price_display = (
                    f"₹{lakh_price / 100:.2f} crore"
                )
            else:
                price_display = f"₹{lakh_price:.2f} lakh"

            st.success("Prediction generated successfully!")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    label="Estimated Price",
                    value=price_display
                )

            with result_col2:
                st.metric(
                    label="Price per sq ft",
                    value=f"₹{predicted_price / area:,.0f}"
                )

            st.caption(
                "This is a machine-learning estimate, not a "
                "verified valuation or guaranteed selling price."
            )

            with st.expander("View submitted property details"):
                st.dataframe(
                    input_df,
                    use_container_width=True,
                    hide_index=True
                )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
