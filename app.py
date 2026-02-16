import streamlit as st
import pandas as pd
import joblib
import os

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🕺",
    layout="wide"
)

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    model = joblib.load("har_model.pkl")
    feature_names = joblib.load("features.pkl")
    return model, feature_names

model, feature_names = load_model()

# -------------------- Header --------------------
st.title("🕺 Human Activity Recognition App")

st.write(
    "Predict human activities using sensor data. "
    "A sample dataset loads automatically when the app starts."
)

# -------------------- About App --------------------
st.markdown("### 📘 About This App")

st.info("""
This application performs **Human Activity Recognition (HAR)** using machine learning.

🔹 The model is trained using **UCI Sensor Data (Human Activity Recognition Dataset)**.  
🔹 Sensor signals such as accelerometer and gyroscope readings are used to classify human activities.  
🔹 The system predicts activities like walking, sitting, standing, running, etc.

### ⚙️ How it works
1. Sensor-based features are extracted from motion data.
2. A trained ML model analyzes the feature patterns.
3. The model predicts the most likely human activity.

📚 **Dataset Source:** UCI Machine Learning Repository  
🎓 Educational and demonstration purpose only.
""")

# -------------------- Risk Mapping --------------------
def map_risk(activity):
    activity = str(activity).lower()

    no_risk = ["sitting", "standing", "lying"]
    neutral = ["walking", "walking_upstairs", "walking_downstairs"]
    high_risk = ["running", "jumping", "fall", "sudden_movement"]

    if activity in no_risk:
        return "No Risk"
    elif activity in neutral:
        return "Neutral"
    else:
        return "High Risk"

# -------------------- Auto Load Sample --------------------
DATA_FILE = "sample_test.csv"
data = None

if os.path.exists(DATA_FILE):
    data = pd.read_csv(DATA_FILE)
    st.success("✅ Loaded sample_test.csv automatically")
else:
    st.warning("⚠️ sample_test.csv not found in project folder.")

# -------------------- Upload Option --------------------
uploaded_file = st.file_uploader(
    "Upload your own CSV (optional — overrides sample)",
    type=["csv"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("📂 Custom CSV uploaded successfully!")

# -------------------- Main Logic --------------------
if data is not None:

    st.subheader("📄 Preview of Input Data")
    st.dataframe(data.head(), use_container_width=True)

    # Check required features
    if set(feature_names).issubset(data.columns):

        X_test = data[feature_names]

        # Predictions
        predictions = model.predict(X_test)
        data["Predicted Activity"] = predictions

        # Risk Mapping
        data["Risk Level"] = data["Predicted Activity"].apply(map_risk)

        st.subheader("🎯 Prediction Results")
        st.success("Predictions completed successfully!")

        st.dataframe(data.head(), use_container_width=True)

        # -------------------- Activity Distribution --------------------
        st.subheader("📊 Activity Distribution")
        activity_counts = data["Predicted Activity"].value_counts()
        st.bar_chart(activity_counts)

        # -------------------- Risk Distribution --------------------
        st.subheader("⚠️ Risk Level Distribution")
        st.bar_chart(data["Risk Level"].value_counts())

        # -------------------- Insights --------------------
        st.subheader("📈 Insights")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Samples", len(data))

        with col2:
            st.metric(
                "Unique Activities",
                data["Predicted Activity"].nunique()
            )

        with col3:
            most_common = activity_counts.idxmax()
            st.metric("Most Common Activity", most_common)

        # -------------------- Download Results --------------------
        st.subheader("⬇️ Download Results")

        csv = data.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name="predicted_activities.csv",
            mime="text/csv"
        )

    else:
        missing = list(set(feature_names) - set(data.columns))
        st.error("❌ Uploaded data is missing required features.")
        st.write("Missing columns:", missing)

else:
    st.info("Upload a CSV or place sample_test.csv in project folder.")
