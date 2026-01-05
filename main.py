import streamlit as st

# ✅ MUST be the first Streamlit command
st.set_page_config(
    page_title="Student Dropout Risk Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ imports ------------------
import pandas as pd
import joblib
import os
from dotenv import load_dotenv
import google.generativeai as genai
import plotly.express as px

# ------------------ ENV & API ------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ------------------ MODEL LOADING ------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("xgboost_model.pkl")
    except FileNotFoundError:
        st.error("❌ xgboost_model.pkl not found in project folder")
        return None

model = load_model()
if model is None:
    st.stop()

# ------------------ FEATURE CONFIG ------------------
categorical_cols = [
    'Gender', 'Region', 'Parental_Education',
    'Internet_Access', 'Family_Support',
    'School_Support', 'Activities'
]

expected_columns = [
    'Age', 'Family_Income', 'Distance_from_School',
    'Absences', 'Failures', 'Study_Time_Category',
    'G1', 'G2', 'G3',
    'Gender_Male', 'Region_Urban',
    'Parental_Education_Primary',
    'Parental_Education_Secondary',
    'Internet_Access_Yes',
    'Family_Support_Yes',
    'School_Support_Yes',
    'Activities_Yes'
]

# ------------------ DATA PREP ------------------
def prepare_input_data(**kwargs):
    df = pd.DataFrame([kwargs])
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    return df_encoded[expected_columns]

# ------------------ AI PROMPTS ------------------
def get_prompt(level):
    prompts = {
        "Low Risk": "Provide growth-focused academic and career guidance.",
        "Medium Risk": "Provide targeted academic and emotional intervention.",
        "High Risk": "Provide emergency academic, financial, and psychological support."
    }
    return prompts[level]

# ------------------ UI HEADER ------------------
st.markdown("""
<div style="text-align:center;padding:20px;background:#6c63ff;color:white;border-radius:10px;">
<h1>🎓 Student Dropout Risk Prediction System</h1>
<p>AI-powered early warning & recommendation system</p>
</div>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.header("📋 Student Information")

age = st.sidebar.slider("Age", 15, 22, 17)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
region = st.sidebar.selectbox("Region", ["Rural", "Urban"])
family_income = st.sidebar.number_input("Family Income (₹)", 0, 100000, 15000, step=1000)
parental_education = st.sidebar.selectbox(
    "Parental Education", ["Primary", "Secondary", "Higher Education"]
)
distance_from_school = st.sidebar.slider("Distance from School (km)", 0.0, 50.0, 5.0)
absences = st.sidebar.number_input("Absences", 0, 100, 5)
failures = st.sidebar.number_input("Failures", 0, 10, 0)
study_time_category = st.sidebar.selectbox("Study Time Category", [1, 2, 3, 4])
g1 = st.sidebar.slider("G1", 0, 20, 10)
g2 = st.sidebar.slider("G2", 0, 20, 10)
g3 = st.sidebar.slider("G3", 0, 20, 10)
internet_access = st.sidebar.selectbox("Internet Access", ["Yes", "No"])
family_support = st.sidebar.selectbox("Family Support", ["Yes", "No"])
school_support = st.sidebar.selectbox("School Support", ["Yes", "No"])
activities = st.sidebar.selectbox("Activities", ["Yes", "No"])

# ------------------ PREDICTION ------------------
st.markdown("---")
if st.button("🔮 Predict Dropout Risk", use_container_width=True):

    input_df = prepare_input_data(
        Age=age,
        Family_Income=family_income,
        Distance_from_School=distance_from_school,
        Absences=absences,
        Failures=failures,
        Study_Time_Category=study_time_category,
        G1=g1, G2=g2, G3=g3,
        Gender=gender,
        Region=region,
        Parental_Education=parental_education,
        Internet_Access=internet_access,
        Family_Support=family_support,
        School_Support=school_support,
        Activities=activities
    )

    pred_class = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0]

    risk_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    risk = risk_map[pred_class]
    confidence = max(pred_proba) * 100

    st.subheader("🎯 Prediction Result")

    if risk == "High Risk":
        st.error(f"🚨 {risk} ({confidence:.1f}%)")
    elif risk == "Medium Risk":
        st.warning(f"⚠️ {risk} ({confidence:.1f}%)")
    else:
        st.success(f"✅ {risk} ({confidence:.1f}%)")

    prob_df = pd.DataFrame({
        "Risk Level": ["Low Risk", "Medium Risk", "High Risk"],
        "Probability": pred_proba
    })

    fig = px.bar(prob_df, x="Risk Level", y="Probability", title="Risk Probability")
    st.plotly_chart(fig, use_container_width=True)

    # ------------------ AI RECOMMENDATIONS ------------------
    if GEMINI_API_KEY:
        with st.spinner("Generating AI recommendations..."):
            model_gen = genai.GenerativeModel("gemini-2.0-flash")
            response = model_gen.generate_content(
                f"Student is at {risk}. {get_prompt(risk)}"
            )
            st.subheader("🤖 AI Recommendations")
            st.markdown(response.text)
    else:
        st.info("ℹ️ Gemini API key not found. AI recommendations disabled.")

# ------------------ FOOTER ------------------
st.markdown("""
<hr>
<div style="text-align:center;color:gray;">
Built with ❤️ using Streamlit, XGBoost & Gemini AI
</div>
""", unsafe_allow_html=True)
