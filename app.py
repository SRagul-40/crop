import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EcoHarvest AI | Smart Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS (DESIGN SYSTEM)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    /* GLOBAL RESET & BACKGROUND */
    .stApp {
        background: linear-gradient(135deg, #051405 0%, #0d2b0d 100%);
        font-family: 'Helvetica Neue', sans-serif;
    }

    /* TEXT COLORS */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #e8f5e9 !important; /* Pale Green Text */
    }
    
    /* SPECIAL HIGHLIGHTS */
    .highlight-text {
        color: #FFD700 !important; /* Gold */
        font-weight: bold;
    }

    /* INPUT CARDS */
    .input-card {
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #2e7d32;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    /* HEADERS */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: -webkit-linear-gradient(#81c784, #2e7d32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #a5d6a7 !important;
        margin-bottom: 40px;
    }

    /* INPUT WIDGETS STYLING */
    .stNumberInput > label, .stSlider > label {
        font-size: 1.1rem !important;
        color: #FFD700 !important; /* Gold Labels */
        font-weight: 600;
    }
    
    /* BUTTON STYLING */
    .stButton>button {
        background: linear-gradient(90deg, #2e7d32 0%, #1b5e20 100%);
        color: #e8f5e9;
        font-size: 1.5rem;
        font-weight: bold;
        padding: 15px 30px;
        border-radius: 12px;
        border: 2px solid #81c784;
        width: 100%;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        color: #FFD700;
        border-color: #FFD700;
    }

    /* RESULT BOX */
    .result-container {
        background-color: #1b381b;
        border-left: 10px solid #FFD700;
        padding: 30px;
        border-radius: 10px;
        text-align: center;
        margin-top: 30px;
        animation: fadeIn 1.5s;
    }
    
    .result-value {
        font-size: 4rem;
        color: #FFD700 !important;
        font-weight: 900;
        text-shadow: 0px 0px 20px rgba(255, 215, 0, 0.3);
    }

    /* HIDE STREAMLIT BRANDING */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. SMART MODEL LOADER (AUTO-TRAINS IF MISSING)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_model_data():
    filename = "crop_yield_model.pkl"
    
    # Check if file exists
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    
    # If not found, train it now (Lazy Training)
    else:
        status_container = st.empty()
        status_container.warning("⚙️ First-time setup: Downloading data & training model... (approx. 20s)")
        
        try:
            import kagglehub
            
            # 1. Download Data
            path = kagglehub.dataset_download("yaminh/crop-yield-prediction")
            file_path = os.path.join(path, "crop yield data sheet.xlsx")
            
            # Fallback search
            if not os.path.exists(file_path):
                for f in os.listdir(path):
                    if f.endswith(".xlsx"):
                        file_path = os.path.join(path, f)
                        break
            
            # 2. Load & Clean
            cy = pd.read_excel(file_path)
            cy = cy.dropna()
            
            # 3. Encode Temperature
            LE = LabelEncoder()
            # Ensure column exists and handle it
            if 'Temperatue' in cy.columns: # Note: dataset typo 'Temperatue'
                cy['Temperatue'] = cy['Temperatue'].astype(str)
                cy['Temperatue'] = LE.fit_transform(cy['Temperatue'])
            
            # 4. Train
            # Cols: Rain Fall (mm), Fertilizer, Temperatue, Nitrogen (N), Phosphorus (P), Potassium (K)
            ind = cy[["Rain Fall (mm)", "Fertilizer", "Temperatue", "Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]]
            dep = cy["Yeild (Q/acre)"]
            
            LR = LinearRegression()
            LR.fit(ind, dep)
            
            # 5. Save & Return
            data = {"model": LR, "encoder_temp": LE}
            with open(filename, "wb") as f:
                pickle.dump(data, f)
            
            status_container.success("✅ System Ready!")
            time.sleep(1)
            status_container.empty()
            return data
            
        except Exception as e:
            st.error(f"Initialization Error: {e}")
            return None

data = get_model_data()

# -----------------------------------------------------------------------------
# 4. HEADER SECTION
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🌾 EcoHarvest AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Crop Yield Forecasting System</div>', unsafe_allow_html=True)

if data:
    model = data["model"]
    temp_encoder = data.get("encoder_temp")

    # -------------------------------------------------------------------------
    # 5. INPUT FORM
    # -------------------------------------------------------------------------
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    
    # Row 1: Environmental Factors
    st.markdown("### <span class='highlight-text'>1. Environmental Parameters</span>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rain = st.number_input("Rainfall (mm)", min_value=0.0, value=1200.0, step=10.0, format="%.1f")
    with col2:
        temp_input = st.number_input("Temperature (°C)", min_value=0, max_value=60, value=28, step=1)
    with col3:
        fertilizer = st.number_input("Fertilizer Usage (kg/acre)", min_value=0.0, value=75.0, step=1.0)

    st.markdown("---")

    # Row 2: Soil Nutrients
    st.markdown("### <span class='highlight-text'>2. Soil Composition (NPK)</span>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    with c1:
        nitrogen = st.slider("Nitrogen (N)", 0.0, 100.0, 80.0)
    with c2:
        phosphorus = st.slider("Phosphorus (P)", 0.0, 100.0, 24.0)
    with c3:
        potassium = st.slider("Potassium (K)", 0.0, 100.0, 20.0)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 6. PREDICTION LOGIC
    # -------------------------------------------------------------------------
    
    col_spacer1, col_btn, col_spacer2 = st.columns([1, 2, 1])
    
    with col_btn:
        predict_btn = st.button("CALCULATE YIELD")

    if predict_btn:
        with st.spinner("Processing satellite & soil data..."):
            time.sleep(0.8)
            
            # Encode Temperature Input
            final_temp = 0
            if temp_encoder:
                try:
                    # Match input to training labels
                    final_temp = temp_encoder.transform([str(int(temp_input))])[0]
                except:
                    # Fallback for unseen values to prevent crash
                    final_temp = 0 
            else:
                final_temp = temp_input

            # Predict
            try:
                inputs = np.array([[rain, fertilizer, final_temp, nitrogen, phosphorus, potassium]])
                prediction = model.predict(inputs)
                yield_val = prediction[0]
                
                # Display
                st.markdown(f"""
                <div class="result-container">
                    <h3>PREDICTED YIELD</h3>
                    <div class="result-value">{yield_val:.2f} Q/acre</div>
                    <p style="color: #a5d6a7; margin-top: 10px;">Based on input parameters</p>
                </div>
                """, unsafe_allow_html=True)
                
                if yield_val > 10:
                    st.balloons()
            
            except Exception as e:
                st.error(f"Calculation Error: {e}")

# -----------------------------------------------------------------------------
# 7. FOOTER
# -----------------------------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #558b2f; font-size: 0.9rem;">
    EcoHarvest Systems &copy; 2024 | Powered by Machine Learning
</div>
""", unsafe_allow_html=True)
