# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 15:17:35 2026
@author: Thilak

PROJECT:
--------
Health Predictor Web App using Machine Learning
(Diabetes | Heart Disease | Parkinson’s)

FRAMEWORK:
----------
Streamlit + Pickle Models
"""
# =========================================================
# 📦 IMPORTS
# =========================================================
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# =========================================================
# 🧠 LOAD ML MODELS
# =========================================================
diabetes_model = pickle.load(open("diabetes_model.sav", "rb"))
heart_disease_model = pickle.load(open("heart_disease_model.sav", "rb"))
parkinsons_model = pickle.load(open("parkinsons_model.sav", "rb"))
# =========================================================
# 📚 SIDEBAR
# =========================================================
with st.sidebar:  

    #---------- Sidebar Header Section ----------
    st.markdown(
        """
        <div class="sidebar-box sidebar-box1">
            <h2>🩺 Health Predictor</h2>
            <p>Disease Detection by ML</p>
            <p>Project Coordinator</p>
            <p>Prof. Saravanan</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.divider() 

    #---------- Sidebar Menu Section ----------
    selected = option_menu(
        "Select Prediction",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction"],
        icons=[
            "droplet-half",      # Diabetes
            "heart-fill",        # Heart
            "person-lines-fill"  # Parkinson's
        ],
        default_index=0,
    )
    st.divider()    

    #---------- Sidebar Footer Section ----------
    st.markdown(
        """
        <div class="sidebar-box sidebar-box3">
            <h3>THILAK S</h3>
            <p>U25PG507DTS041</p>
            <p>1<sup>st</sup> Year MSc Data Science</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    #---------- Sidebar Footer Fixed ----------
    st.markdown(
        """
        <div class="sidebar-footer">
            <small>© 2026 Health AI</small>
        </div>
        """,
        unsafe_allow_html=True
    )


#------------ Mmain Content Section  Start--------------------
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# =========================================================
# 🩸 DIABETES PREDICTION MODULE
# =========================================================
if (selected == 'Diabetes Prediction'):     
    # -----------------------------------------------------
    # 1️⃣ SESSION STATE INITIALIZATION (Default Values)
    # -----------------------------------------------------
    # These defaults ensure the form retains values
    # and can be reset or auto-filled safely
    defaults = {
        "Pregnancies": 0, "Glucose": 0, "BloodPressure": 0,
        "SkinThickness": 0, "Insulin": 0,  "BMI": 0.0,
        "DPF": 0.0,  "Age": 1
    }

    # Initialize session state keys if not already present
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # -----------------------------------------------------
    # 2️⃣ CLEAR FORM FUNCTION
    # -----------------------------------------------------
    # Resets all input fields back to default values
    def clear_form():
        for key, value in defaults.items():
            st.session_state[key] = value

    # -----------------------------------------------------
    # 3️⃣ SAMPLE PATIENT DATA (FOR DEMO PURPOSE)
    # -----------------------------------------------------
    # Helps users quickly test the model with realistic data
    DIABETES_SAMPLES = {
        "Select Sample": None,
    
        "1️⃣ Sample Data": {
            "Pregnancies": 3, "Glucose": 126, "BloodPressure": 88,
            "SkinThickness": 41, "Insulin": 235, "BMI": 39.3,
            "DPF": 0.704, "Age": 27
        },
    
        "2️⃣ Sample Data": {
            "Pregnancies": 2, "Glucose": 135, "BloodPressure": 82,
            "SkinThickness": 28, "Insulin": 140, "BMI": 28.9,
            "DPF": 0.78, "Age": 45
        },
    
        "3️⃣ Sample Data": {
            "Pregnancies": 6, "Glucose": 178, "BloodPressure": 90,
            "SkinThickness": 35, "Insulin": 220, "BMI": 34.6,
            "DPF": 1.45, "Age": 62
        }
    }

    # -----------------------------------------------------
    # 4️⃣ APPLY SELECTED SAMPLE DATA
    # -----------------------------------------------------
    # Copies selected sample values into session state
    def apply_diabetes_sample(sample_name):
        sample = DIABETES_SAMPLES.get(sample_name)
        if sample:
            for key, value in sample.items():
                st.session_state[key] = value
  
    # -----------------------------------------------------
    # 5️⃣ PAGE HEADER & ACTION BUTTONS
    # -----------------------------------------------------
    col_title, col_btn1, col_btn2 = st.columns([4, 1, 1])

    with col_title:
        st.header("🩸 Diabetes Prediction", divider="blue")

    with col_btn1:
        st.markdown("<br>", unsafe_allow_html=True)
           
    with col_btn2:      
        st.button("🧹 Clear", type="secondary", on_click=clear_form)

    # -----------------------------------------------------
    # 6️⃣ SAMPLE SELECTION DROPDOWN
    # -----------------------------------------------------
    sample_choice = st.selectbox(
        "🧪 Load Sample Patient",
        list(DIABETES_SAMPLES.keys()),
        index=0
    )

    # Load sample data when selected
    if sample_choice != "Select Sample":
        apply_diabetes_sample(sample_choice)
           
    # -----------------------------------------------------
    # 7️⃣ DIABETES INPUT FORM
    # -----------------------------------------------------
    with st.form("diabetes_form"):

        # --- Row 1 ---
        col1, col2, col3 = st.columns(3)
        with col1:
            Pregnancies = st.number_input(
                "Number of Pregnancies", 0, 20, key="Pregnancies"
            )
        with col2:
            Glucose = st.number_input(
                "Glucose Level (mg/dL)", 0, 300, key="Glucose"
            )
        with col3:
            BloodPressure = st.number_input(
                "Blood Pressure (mm Hg)", 0, 200, key="BloodPressure"
            )

        # --- Row 2 ---
        col1, col2, col3 = st.columns(3)
        with col1:
            SkinThickness = st.number_input(
                "Skin Thickness (mm)", 0, 100, key="SkinThickness"
            )
        with col2:
            Insulin = st.number_input(
                "Insulin Level (µU/mL)", 0, 900, key="Insulin"
            )
        with col3:
            BMI = st.number_input(
                "BMI", 0.0, 70.0, format="%.2f", key="BMI"
            )

        # --- Row 3 ---
        col1, col2 = st.columns(2)
        with col1:
            DPF = st.number_input(
                "Diabetes Pedigree Function", 0.0, 3.0, format="%.3f", key="DPF"
            )
        with col2:
            Age = st.number_input(
                "Age", 1, 120, key="Age"
            )
    
        # -------------------------------------------------
        # 8️⃣ PREDICTION BUTTON
        # -------------------------------------------------
        col1, col2 = st.columns(2)
        with col1:
            predict_btn = st.form_submit_button("🔍 Diabetes Test Result", type="primary")        

    # -----------------------------------------------------
    # 9️⃣ DIABETES PREDICTION & VALIDATION
    # -----------------------------------------------------
    if predict_btn:
        
        # -------------------------------------------------
        # 9.1️⃣ BASIC INPUT VALIDATION
        # -------------------------------------------------
        # Collects warnings for unrealistic or unsafe inputs
        errors = []
    
        if Glucose < 70:
            errors.append("⚠️ Glucose level seems too low.")
        if BloodPressure < 40:
            errors.append("⚠️ Blood Pressure seems too low.")
        if BMI < 10:
            errors.append("⚠️ BMI value seems invalid.")
        if Age < 10:
            errors.append("⚠️ Age must be at least 10 years.")

        # Display validation errors (if any)
        if errors:
            st.error("Please correct the following:")
            for err in errors:
                st.write(err)
        # -------------------------------------------------
        # 9.2️⃣ MODEL PREDICTION
        # -------------------------------------------------
        else:
            # Predict diabetes outcome (0 = No, 1 = Yes)
            diab_prediction = diabetes_model.predict([[
                Pregnancies, Glucose, BloodPressure,
                SkinThickness, Insulin, BMI, DPF, Age
            ]])

            # Display prediction result
            if diab_prediction[0] == 1:
                st.error("🔴 The person is Diabetic")
            else:
                st.success("🟢 The person is not Diabetic")

            # ------------------------------------------------- 
            # 9.3️⃣ RISK PROBABILITY CALCULATION
            # -------------------------------------------------
            # Use probability if model supports it
            if hasattr(diabetes_model, "predict_proba"):
                proba = diabetes_model.predict_proba([[
                    Pregnancies, Glucose, BloodPressure,
                    SkinThickness, Insulin, BMI, DPF, Age
                ]])
                risk = proba[0][1] * 100
            # Fallback for models without probability support
            else:
                prediction = diabetes_model.predict([[
                    Pregnancies, Glucose, BloodPressure,
                    SkinThickness, Insulin, BMI, DPF, Age
                ]])
                risk = 100 if prediction[0] == 1 else 0

            # -------------------------------------------------
            # 9.4️⃣ RISK VISUALIZATION      
            # -------------------------------------------------           
            st.subheader("📊 Diabetes Risk Probability")
            
            st.metric("Risk of Diabetes", f"{risk:.2f} %")
            st.progress(int(risk))

            # -------------------------------------------------
            # 9.5️⃣ RISK CATEGORY INTERPRETATION
            # -------------------------------------------------
            if risk >= 70:
                st.error("🔴 High Risk of Diabetes")
            elif risk >= 40:
                st.warning("🟠 Moderate Risk — lifestyle changes advised")
            else:
                st.success("🟢 Low Risk: The person is not Diabetic")
                

# =========================================================
# ❤️ HEART DISEASE PREDICTION MODULE
# =========================================================
if selected == 'Heart Disease Prediction':

    # -----------------------------------------------------
    # 1️⃣ SESSION STATE DEFAULT VALUES
    # -----------------------------------------------------
    # Default values ensure form persistence and reset safety
    heart_defaults = {
        "age": 1, "sex": 0, "cp": 0, "trestbps": 80,  
        "chol": 100, "fbs": 0, "restecg": 0, "thalach": 60, 
        "exang": 0, "oldpeak": 0.0, "slope": 0, "ca": 0, "thal": 0
    }

    # Initialize session state
    for k, v in heart_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
            
    # -----------------------------------------------------
    # 2️⃣ CLEAR FORM FUNCTION
    # -----------------------------------------------------
    # Resets all heart disease inputs to default values
    def clear_heart_form():
        for k, v in heart_defaults.items():
            st.session_state[k] = v
    
    # -----------------------------------------------------
    # 3️⃣ SAMPLE DATA (FOR QUICK TESTING)
    # -----------------------------------------------------
    # Used to auto-fill realistic patient profiles
    HEART_SAMPLES = {
        "Select Sample": None,
    
        "1️⃣ Sample Data": {
            "age": 25, "sex": 0, "cp": 0,  "trestbps": 108,
            "chol": 165, "fbs": 0, "restecg": 0, "thalach": 190,
            "exang": 0, "oldpeak": 0.0,  "slope": 1, "ca": 0, "thal": 0
        },
    
        "2️⃣ Sample Data": {
            "age": 52, "sex": 1, "cp": 2, "trestbps": 138,
            "chol": 245, "fbs": 0, "restecg": 1, "thalach": 150,
            "exang": 0, "oldpeak": 1.3, "slope": 1, "ca": 0, "thal": 1
        },
    
        "3️⃣ Sample Data": {
            "age": 67, "sex": 1, "cp": 3, "trestbps": 168,
            "chol": 295, "fbs": 1,
            "restecg": 2, "thalach": 118, "exang": 1,
            "oldpeak": 2.9, "slope": 2, "ca": 2, "thal": 2
        }
    }

    # -----------------------------------------------------
    # 4️⃣ APPLY SELECTED SAMPLE
    # -----------------------------------------------------
    # Loads chosen sample data into session state
    def apply_heart_sample(sample_name):
        sample = HEART_SAMPLES.get(sample_name)
        if sample:
            for key, value in sample.items():
                st.session_state[key] = value

    # -----------------------------------------------------
    # 5️⃣ PAGE HEADER & ACTION BUTTONS
    # -----------------------------------------------------
    col_title, col_btn1, col_btn2 = st.columns([4, 1, 1])

    with col_title:
        st.header("❤️ Heart Disease Prediction", divider="red")
    
    with col_btn1:
        st.markdown("<br>", unsafe_allow_html=True)
            
    with col_btn2:
        st.button("🧹 Clear", on_click=clear_heart_form)

    # -----------------------------------------------------
    # 6️⃣ SAMPLE SELECTION DROPDOWN
    # -----------------------------------------------------
    sample_choice = st.selectbox(
        "🧪 Load Sample Patient",
        list(HEART_SAMPLES.keys()),
        index=0
    )
    
    if sample_choice != "Select Sample":
        apply_heart_sample(sample_choice)

    # -----------------------------------------------------
    # 7️⃣ HEART DISEASE INPUT FORM
    # -----------------------------------------------------
    with st.form("heart_form"):

        # --- Row 1 ---
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input('Age', 1, 120, key="age")
        with col2:
            sex = st.number_input('Sex (1 = Male, 0 = Female)', 0, 1, key="sex")
        with col3:
            cp = st.number_input('Chest Pain Type (0–3)', 0, 3, key="cp")

        # --- Row 2 ---
        col1, col2, col3 = st.columns(3)
        with col1:
            trestbps = st.number_input('Resting Blood Pressure', 80, 200, key="trestbps")
        with col2:
            chol = st.number_input('Serum Cholesterol', 100, 600, key="chol")
        with col3:
            fbs = st.number_input('Fasting Blood Sugar > 120', 0, 1, key="fbs")

        # --- Row 3 ---
        col1, col2, col3 = st.columns(3)
        with col1:
            restecg = st.number_input('Resting ECG (0–2)', 0, 2, key="restecg")
        with col2:
            thalach = st.number_input('Max Heart Rate', 60, 250, key="thalach")
        with col3:
            exang = st.number_input('Exercise Induced Angina', 0, 1, key="exang")

        # --- Row 4 ---
        col1, col2, col3 = st.columns(3)
        with col1:
            oldpeak = st.number_input('ST Depression', 0.0, 10.0, key="oldpeak")
        with col2:
            slope = st.number_input('Slope (0–2)', 0, 2, key="slope")
        with col3:
            ca = st.number_input('Major Vessels', 0, 4, key="ca")

        thal = st.number_input('Thal (0–2)', 0, 2, key="thal")

        predict_btn = st.form_submit_button("🔍 Heart Disease Test Result", type="primary")

    # -----------------------------------------------------
    # 8️⃣ PREDICTION & VALIDATION
    # -----------------------------------------------------
    if predict_btn:

        # --- Input Validation ---
        errors = []

        if age < 10:
            errors.append("⚠️ Age must be at least 10 years.")
        if trestbps < 80 or trestbps > 200:
            errors.append("⚠️ Resting BP must be 80–200 mm Hg.")
        if chol < 100 or chol > 600:
            errors.append("⚠️ Cholesterol must be 100–600.")
        if thalach < 60 or thalach > 250:
            errors.append("⚠️ Max heart rate must be 60–250.")
        if ca < 0 or ca > 4:
            errors.append("⚠️ Major vessels must be 0–4.")

        # Display validation errors
        if errors:
            st.error("Please correct the following:")
            for e in errors:
                st.write(e)
        # -------------------------------------------------
        # 9️⃣ MODEL INFERENCE & RISK ANALYSIS
        # -------------------------------------------------
        else:
            input_data = [[
                age, sex, cp, trestbps, chol, fbs,
                restecg, thalach, exang, oldpeak,
                slope, ca, thal
            ]]

            # Probability prediction
            proba = heart_disease_model.predict_proba(input_data)
            risk = proba[0][1] * 100   # Probability of disease
            safe = proba[0][0] * 100
    
            # -------------------------------------------------
            # 🔟 RESULT VISUALIZATION
            # -------------------------------------------------
            st.subheader("📊 Risk Assessment")
    
            st.metric(
                label="Heart Disease Risk",
                value=f"{risk:.2f} %",
                delta=f"{safe:.2f} % Healthy"
            )
    
            st.progress(int(risk))

            # -------------------------------------------------
            # 1️⃣1️⃣ RISK CATEGORY INTERPRETATION
            # -------------------------------------------------
            if risk >= 70:
                st.error("🔴 High Risk of Heart Disease")
            elif risk >= 40:
                st.warning("🟠 Moderate Risk — medical consultation advised")
            else:
                st.success("🟢 Low Risk Detected")




# =========================================================
# 🧠 PARKINSON’S DISEASE PREDICTION MODULE
# =========================================================
if (selected == 'Parkinsons Prediction'):
    
    # -----------------------------------------------------
    # 1️⃣ SESSION STATE DEFAULT VALUES
    # -----------------------------------------------------
    # Default initialization for all Parkinson's voice features
    parkinsons_defaults = {
        "fo": 0.0, "fhi": 0.0, "flo": 0.0,
        "Jitter_percent": 0.0, "Jitter_Abs": 0.0,
        "RAP": 0.0, "PPQ": 0.0, "DDP": 0.0,
        "Shimmer": 0.0, "Shimmer_dB": 0.0,
        "APQ3": 0.0, "APQ5": 0.0, "APQ": 0.0, "DDA": 0.0,
        "NHR": 0.0, "HNR": 0.0,
        "RPDE": 0.0, "DFA": 0.0,
        "spread1": 0.0, "spread2": 0.0,
        "D2": 0.0, "PPE": 0.0
    }

    # Initialize session state keys
    for key, val in parkinsons_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # -----------------------------------------------------
    # 2️⃣ CLEAR FORM FUNCTION
    # -----------------------------------------------------
    def clear_parkinsons_form():
        for key, val in parkinsons_defaults.items():
            st.session_state[key] = val

    # -----------------------------------------------------
    # 3️⃣ SAMPLE VOICE DATA (FOR DEMONSTRATION)
    # -----------------------------------------------------
    # Helps users test model using realistic voice patterns
    PARKINSONS_SAMPLES = {
        "Select Sample": None,
    
        "1️⃣ Sample Data": {
            "fo": 120.0, "fhi": 150.0, "flo": 100.0,
            "Jitter_percent": 0.003, "Jitter_Abs": 0.00002,
            "RAP": 0.0015, "PPQ": 0.002, "DDP": 0.004,
            "Shimmer": 0.015, "Shimmer_dB": 0.15,
            "APQ3": 0.008, "APQ5": 0.009, "APQ": 0.012, "DDA": 0.024,
            "NHR": 0.02, "HNR": 25.0,
            "RPDE": 0.35, "DFA": 0.60,
            "spread1": -6.0, "spread2": 0.15,
            "D2": 2.1, "PPE": 0.08
        },
    
        "2️⃣ Sample Data": {
            "fo": 197.076, "fhi": 206.896, "flo":192.055,
            "Jitter_percent": .00289, "Jitter_Abs": 0.00001,
            "RAP": 0.00166, "PPQ": 0.00168, "DDP": 0.00498,
            "Shimmer":  0.01098, "Shimmer_dB": 0.097,
            "APQ3": 0.00563, "APQ5": 0.0068, "APQ": 0.00802, "DDA": 0.01689,
            "NHR": 0.00339, "HNR": 26.775,
            "RPDE": 0.422229, "DFA": 0.741367,
            "spread1": -7.3483, "spread2": 0.177551,
            "D2": 1.743867, "PPE": 0.085569           
        },
    
        "3️⃣ Sample Data": {
            "fo": 165.0, "fhi": 220.0, "flo": 90.0,
            "Jitter_percent": 0.012, "Jitter_Abs": 0.00012,
            "RAP": 0.006, "PPQ": 0.008, "DDP": 0.018,
            "Shimmer": 0.060, "Shimmer_dB": 0.55,
            "APQ3": 0.030, "APQ5": 0.035, "APQ": 0.045, "DDA": 0.090,
            "NHR": 0.08, "HNR": 10.5,
            "RPDE": 0.68, "DFA": 0.78,
            "spread1": -3.5, "spread2": 0.32,
            "D2": 3.4, "PPE": 0.32
        },

        "4️⃣ Sample Data": {
            "fo": 198.383, "fhi": 215.203, "flo": 193.104,
            "Jitter_percent": 0.00212, "Jitter_Abs": 0.00001,
            "RAP": 0.00113, "PPQ": 0.00135, "DDP": 0.00339,
            "Shimmer": 0.01263, "Shimmer_dB": 0.111,
            "APQ3": 0.0064, "APQ5": 0.00825, "APQ": 0.00951, "DDA": 0.01919,
            "NHR": 0.00119, "HNR": 30.775,
            "RPDE": 0.465946, "DFA": 0.738703,
            "spread1": -7.067931, "spread2": 0.175181,
            "D2": 1.512275, "PPE": 0.09632
        }
    }

    # -----------------------------------------------------
    # 4️⃣ APPLY SELECTED SAMPLE DATA
    # -----------------------------------------------------
    # Loads selected voice sample into the form
    def apply_parkinsons_sample(sample_name):
        sample = PARKINSONS_SAMPLES.get(sample_name)
        if sample:
            for key, value in sample.items():
                st.session_state[key] = value


    # -----------------------------------------------------
    # 5️⃣ PAGE HEADER & CLEAR BUTTON
    # -----------------------------------------------------
    col_title, col_btn1, col_btn2 = st.columns([4, 1, 1])
    
    with col_title:
        st.header("🧠 Parkinson’s Prediction", divider="violet")

    with col_btn1:
        st.markdown("<br>", unsafe_allow_html=True)

    with col_btn2:
        st.button("🧹 Clear", type="secondary", on_click=clear_parkinsons_form)

    # -----------------------------------------------------
    # 6️⃣ SAMPLE SELECTION DROPDOWN
    # -----------------------------------------------------
    sample_choice = st.selectbox(
        "🧪 Load Sample Voice Data",
        list(PARKINSONS_SAMPLES.keys()),
        index=0
    )
    
    if sample_choice != "Select Sample":
        apply_parkinsons_sample(sample_choice)    

    # -----------------------------------------------------
    # 7️⃣ PARKINSON’S INPUT FORM      
    # -----------------------------------------------------
    with st.form("parkinsons_form"):

        # --- Frequency & Jitter Metrics ---
        cols = st.columns(5)
        with cols[0]: fo = st.number_input("MDVP:Fo(Hz)", step=0.001, format="%.3f", key="fo")
        with cols[1]: fhi = st.number_input("MDVP:Fhi(Hz)", step=0.001, format="%.3f", key="fhi")
        with cols[2]: flo = st.number_input("MDVP:Flo(Hz)", step=0.001, format="%.3f", key="flo")
        with cols[3]: Jitter_percent = st.number_input("MDVP:Jitter(%)", step=0.00001, format="%.5f", key="Jitter_percent")
        with cols[4]: Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", step=0.00001, format="%.5f", key="Jitter_Abs")

        # --- Jitter & Shimmer Features ---
        cols = st.columns(5)
        with cols[0]: RAP = st.number_input("MDVP:RAP", step=0.00001, format="%.5f", key="RAP")
        with cols[1]: PPQ = st.number_input("MDVP:PPQ", step=0.00001, format="%.5f", key="PPQ")
        with cols[2]: DDP = st.number_input("Jitter:DDP", step=0.00001, format="%.5f", key="DDP")
        with cols[3]: Shimmer = st.number_input("MDVP:Shimmer", step=0.00001, format="%.5f", key="Shimmer")
        with cols[4]: Shimmer_dB = st.number_input("MDVP:Shimmer(dB)", step=0.001, format="%.3f", key="Shimmer_dB")

        # --- Amplitude Perturbation Measures ---
        cols = st.columns(5)
        with cols[0]: APQ3 = st.number_input("Shimmer:APQ3", step=0.00001, format="%.5f", key="APQ3")
        with cols[1]: APQ5 = st.number_input("Shimmer:APQ5", step=0.00001, format="%.5f", key="APQ5")
        with cols[2]: APQ = st.number_input("MDVP:APQ", step=0.00001, format="%.5f", key="APQ")
        with cols[3]: DDA = st.number_input("Shimmer:DDA", step=0.00001, format="%.5f", key="DDA")
        with cols[4]: NHR = st.number_input("NHR", step=0.00001, format="%.5f", key="NHR")

        # --- Noise & Nonlinear Measures ---
        cols = st.columns(5)
        with cols[0]: HNR = st.number_input("HNR", step=0.001, format="%.3f", key="HNR")
        with cols[1]: RPDE = st.number_input("RPDE", step=0.00001, format="%.6f", key="RPDE")
        with cols[2]: DFA = st.number_input("DFA", step=0.00001, format="%.6f", key="DFA")
        with cols[3]: spread1 = st.number_input("spread1", step=0.00001, format="%.6f", key="spread1")
        with cols[4]: spread2 = st.number_input("spread2", step=0.00001, format="%.6f", key="spread2")

        # --- Complexity Measures ---
        cols = st.columns(2)
        with cols[0]: D2 = st.number_input("D2", step=0.00001, format="%.6f", key="D2")
        with cols[1]: PPE = st.number_input("PPE", step=0.00001, format="%.6f", key="PPE")

        predict_btn = st.form_submit_button("🔍 Parkinson's Test Result", type="primary")

    # -----------------------------------------------------
    # 8️⃣ INPUT VALIDATION & ERROR HANDLING
    # -----------------------------------------------------
    if predict_btn:

        errors = []
    
        # Basic numeric sanity checks
        if fo <= 0 or fhi <= 0 or flo <= 0:
            errors.append("⚠️ Frequency values (Fo, Fhi, Flo) must be greater than 0.")
    
        if Jitter_percent < 0 or Shimmer < 0:
            errors.append("⚠️ Jitter and Shimmer values cannot be negative.")
    
        if HNR <= 0:
            errors.append("⚠️ HNR must be greater than 0.")
    
        if D2 <= 0 or PPE <= 0:
            errors.append("⚠️ D2 and PPE must be greater than 0.")
    
        # Prevent meaningless all-zero input
        all_inputs = [
            fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
            Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA,
            NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE
        ]
    
        if all(value == 0 for value in all_inputs):
            errors.append("⚠️ Please enter valid data. All values cannot be zero.")
    
        # -------------------------------------------------
        # 9️⃣ DISPLAY ERRORS OR PERFORM PREDICTION
        # -------------------------------------------------
        if errors:
            st.error("Please fix the following issues before prediction:")
            for err in errors:
                st.write(err)
    
        # --------- Prediction ----------
        else:
            input_data = [[
                fo, fhi, flo, Jitter_percent, Jitter_Abs,
                RAP, PPQ, DDP, Shimmer, Shimmer_dB,
                APQ3, APQ5, APQ, DDA, NHR, HNR,
                RPDE, DFA, spread1, spread2, D2, PPE
            ]]
    
            prediction = parkinsons_model.predict(input_data)

            # -------------------------------------------------
            # 🔟 RESULT DISPLAY
            # -------------------------------------------------
            if prediction[0] == 1:
                st.error("🔴 Parkinson’s Disease Detected")
            else:
                st.success("🟢 No Parkinson’s Disease Detected")

            prediction = parkinsons_model.predict(input_data)

            # --------------------------------------------
            # 📊 RISK PROBABILITY CALCULATION
            # --------------------------------------------
            if hasattr(parkinsons_model, "predict_proba"):
                proba = parkinsons_model.predict_proba(input_data)
                risk = proba[0][1] * 100
            else:
                risk = 100 if prediction[0] == 1 else 0
            
            # --------------------------------------------
            # 📊 RISK VISUALIZATION
            # --------------------------------------------
            st.subheader("📊 Risk Assessment for Parkinson’s")
            
            st.metric("Parkinson’s Risk", f"{risk:.2f} %")
            st.progress(int(risk))            
            # --------------------------------------------
            # 🚦 RISK CATEGORY INTERPRETATION
            # --------------------------------------------
            if risk >= 70:
                st.error("🔴 High Risk of Parkinson’s Disease")
            elif risk >= 40:
                st.warning("🟠 Moderate Risk — Neurological evaluation advised")
            else:
                st.success("🟢 Low Risk")

#------------ Mmain Content Section End--------------------    
st.markdown('</div>', unsafe_allow_html=True)

