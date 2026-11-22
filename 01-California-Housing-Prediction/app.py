from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource
def load_model(model_path: Path):
    """Load the trained pipeline once and cache it across reruns."""
    return joblib.load(model_path)


st.title("🏠 California Housing Price Predictor")
st.write(
    "Predict median house values in California using a trained "
    "**RobustScaler + Random Forest** pipeline (`RMSE: 0.5122`, `R²: 0.7998`)."
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found! Run `python 01-California-Housing-Prediction/train.py` first."
    )
    st.stop()

model = load_model(MODEL_PATH)

# Sidebar inputs
st.sidebar.header("🏡 Property & District Features")

med_inc = st.sidebar.slider(
    "Median Income ($10k)", min_value=0.5, max_value=15.0, value=3.87, step=0.1
)
house_age = st.sidebar.slider(
    "House Age (years)", min_value=1.0, max_value=52.0, value=28.0, step=1.0
)
ave_rooms = st.sidebar.slider(
    "Average Rooms", min_value=1.0, max_value=15.0, value=5.4, step=0.1
)
ave_bedrms = st.sidebar.slider(
    "Average Bedrooms", min_value=0.5, max_value=5.0, value=1.1, step=0.1
)
population = st.sidebar.number_input(
    "District Population", min_value=10.0, max_value=15000.0, value=1425.0, step=50.0
)
ave_occup = st.sidebar.slider(
    "Average Occupancy (people/house)",
    min_value=1.0,
    max_value=10.0,
    value=3.0,
    step=0.1,
)
latitude = st.sidebar.slider(
    "Latitude", min_value=32.5, max_value=42.0, value=34.05, step=0.05
)
longitude = st.sidebar.slider(
    "Longitude", min_value=-124.3, max_value=-114.3, value=-118.25, step=0.05
)

# بناء DataFrame بنفس أسماء وترتيب الأعمدة لي تدرب عليها الموديل
input_data = pd.DataFrame(
    [
        {
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": latitude,
            "Longitude": longitude,
        }
    ]
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📍 Selected Location")
    st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=6)

with col2:
    st.subheader("💰 Price Prediction")
    st.dataframe(input_data, use_container_width=True, hide_index=True)

    if st.button("Predict House Price", type="primary", use_container_width=True):
        prediction = model.predict(input_data)[0]
        predicted_price = prediction * 100_000

        st.metric(
            label="Estimated Median House Value",
            value=f"${predicted_price:,.0f}",
        )

        if prediction >= 4.95:
            st.info(
                "Note: Target values in this dataset are capped at $500,000, "
                "so the true market price may be higher."
            )
