import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Insurance AI Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR AESTHETICS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---
def apply_feature_engineering(data):
    """Ensures input data has the same bands used during model training."""
    data = data.copy()
    data['age_band'] = pd.cut(data['age'], bins=[0, 30, 45, 60, 100], labels=['18-30', '31-45', '46-60', '60+'])
    data['bmi_band'] = pd.cut(data['bmi'], bins=[0, 25, 30, 35, 100], labels=['normal', 'overweight', 'obese', 'severely_obese'])
    return data

@st.cache_resource
def load_artifacts():
    """Loads model and preprocessor safely."""
    try:
        model = joblib.load('models/gb_model.joblib')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3408/3408591.png", width=100)
st.sidebar.title("Insurance AI")
menu = ["Dashboard & EDA", "Model Diagnostics", "Premium Predictor"]
choice = st.sidebar.selectbox("Action Center", menu)

# --- LOAD DATA ---
if os.path.exists('data/raw/insurance.csv'):
    df = pd.read_csv('data/raw/insurance.csv')
else:
    st.error("Dataset not found! Please ensure data/raw/insurance.csv exists.")
    st.stop()

# --- PAGE 1: DASHBOARD & EDA ---
if choice == "Dashboard & EDA":
    st.title("📊 Dataset Insights")
    
    # Key Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Age", f"{df['age'].mean():.1f} Years")
    m2.metric("Avg BMI", f"{df['bmi'].mean():.1f}")
    m3.metric("Smoker %", f"{(df['smoker'] == 'yes').mean()*100:.1f}%")
    m4.metric("Avg Charge", f"${df['charges'].mean():,.0f}")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Regional Distribution")
        fig1, ax1 = plt.subplots()
        df['region'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1, colors=sns.color_palette("pastel"))
        st.pyplot(fig1)

    with col2:
        st.subheader("Impact of Smoking on Costs")
        fig2, ax2 = plt.subplots()
        sns.boxenplot(x='smoker', y='charges', data=df, palette="magma", ax=ax2)
        st.pyplot(fig2)

# --- PAGE 2: MODEL DIAGNOSTICS ---
elif choice == "Model Diagnostics":
    st.title("🧪 Model Performance Analysis")
    
    if os.path.exists('reports/segment_mae.csv'):
        segments = pd.read_csv('reports/segment_mae.csv')
        
        st.subheader("Error Heatmap (MAE by Segment)")
        pivot_table = segments.pivot(index="age_band", columns="bmi_band", values="mae")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
        st.pyplot(fig)
        
        st.info("💡 **Insight:** Darker areas represent customer profiles where the model is less certain (higher error).")
    else:
        st.warning("No evaluation report found. Run `python -m src.evaluate` first.")

# --- PAGE 3: PREDICTOR ---
elif choice == "Premium Predictor":
    st.title("🔮 AI Premium Estimator")
    model = load_artifacts()
    
    if model:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Customer Profile")
            with st.container():
                c1, c2 = st.columns(2)
                age = c1.slider("Age", 18, 64, 30)
                bmi = c2.slider("BMI", 15.0, 50.0, 25.0)
                
                c3, c4, c5 = st.columns(3)
                smoker = c3.radio("Smoker?", ["yes", "no"])
                children = c4.number_input("Children", 0, 5, 0)
                region = c5.selectbox("Region", df['region'].unique())
                sex = st.selectbox("Sex", df['sex'].unique())

        # Prepare data for prediction
        input_df = pd.DataFrame([[age, sex, bmi, children, smoker, region]], 
                                columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
        input_df = apply_feature_engineering(input_df)
        
        prediction = model.predict(input_df)[0]

        with col2:
            st.markdown("### Prediction Results")
            st.metric("Estimated Premium", f"${prediction:,.2f}")
            
            # Interactive What-If
            if smoker == "yes":
                input_no_smoke = input_df.copy()
                input_no_smoke['smoker'] = "no"
                saving = prediction - model.predict(input_no_smoke)[0]
                st.success(f"🚭 Potential Savings if you quit: **${saving:,.2f}**")
            
            if bmi > 25:
                input_healthy_bmi = input_df.copy()
                input_healthy_bmi['bmi'] = 24.0
                input_healthy_bmi['bmi_band'] = "normal"
                saving_bmi = prediction - model.predict(input_healthy_bmi)[0]
                st.warning(f"🏃 Potential Savings with healthy BMI: **${saving_bmi:,.2f}**")