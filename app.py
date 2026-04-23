# -*- coding: utf-8 -*-
"""
CARDIOVASCULAR RISK ASSESSMENT - Web Application
Flask-based hospital diagnostic system with SHAP explainability
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import json

# ─────────────────────────────────────────────
# INITIALIZE FLASK
# ─────────────────────────────────────────────

app = Flask(__name__)

# ─────────────────────────────────────────────
# LOAD & TRAIN MODEL ON STARTUP
# ─────────────────────────────────────────────

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

FEATURE_NAMES = COLUMNS[:-1]

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

def load_and_train_model():
    """Load data, train model, and setup SHAP explainer"""
    try:
        df = pd.read_csv(URL, header=None, names=COLUMNS, na_values="?")
        df.dropna(inplace=True)
        df["target"] = (df["target"] > 0).astype(int)
        
        X = df[FEATURE_NAMES].values
        y = df["target"].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        explainer = shap.Explainer(model, X_train_scaled)
        
        return model, scaler, explainer, X_train_scaled
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None

# Load model at startup
print("[INFO] Loading AI model and SHAP explainer...")
MODEL, SCALER, EXPLAINER, X_TRAIN = load_and_train_model()
print("[SUCCESS] Model ready!")

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def get_risk_level_info(prob):
    """Determine risk level, color, and recommendations"""
    if prob > 0.7:
        return {
            "level": "HIGH RISK",
            "color": "#d62728",
            "recommendations": [
                "URGENT: Schedule cardiology consultation within 1 week",
                "Undergo comprehensive cardiac evaluation and stress testing",
                "Discuss medication options with cardiovascular specialist",
                "Begin daily moderate exercise (with doctor approval)",
                "Reduce sodium and saturated fat intake"
            ]
        }
    elif prob > 0.5:
        return {
            "level": "MODERATE RISK",
            "color": "#ff7f0e",
            "recommendations": [
                "Schedule doctor appointment within 2-4 weeks",
                "Monitor blood pressure and cholesterol levels regularly",
                "Increase physical activity to 150 min/week",
                "Follow heart-healthy diet (Mediterranean style)",
                "Manage stress through relaxation techniques"
            ]
        }
    elif prob > 0.3:
        return {
            "level": "MODERATE-LOW RISK",
            "color": "#fbc02d",
            "recommendations": [
                "Schedule annual cardiovascular check-up",
                "Monitor blood pressure and cholesterol quarterly",
                "Maintain consistent exercise routine",
                "Keep heart-healthy dietary habits",
                "Track lifestyle changes regularly"
            ]
        }
    else:
        return {
            "level": "LOW RISK",
            "color": "#2ca02c",
            "recommendations": [
                "Continue with regular annual check-ups",
                "Maintain healthy diet and exercise habits",
                "Monitor blood pressure and cholesterol yearly",
                "Avoid smoking and limit alcohol",
                "Manage stress and maintain healthy weight"
            ]
        }

def get_shap_insights(patient_data):
    """Get SHAP-based risk factors and protective factors"""
    x = np.array([patient_data[f] for f in FEATURE_NAMES]).reshape(1, -1)
    x_scaled = SCALER.transform(x)
    
    shap_values = EXPLAINER(x_scaled)
    vals = shap_values.values
    
    if len(vals.shape) == 3 and vals.shape[2] > 1:
        contributions = vals[0, :, 1]
    else:
        contributions = vals[0]
    
    idx = np.argsort(np.abs(contributions))[-8:][::-1]
    
    risk_factors = []
    protective_factors = []
    
    for i in idx:
        f = FEATURE_NAMES[i]
        v = patient_data[f]
        contrib = contributions[i]
        
        if contrib > 0.05:
            risk_factors.append({"feature": f, "value": v, "impact": float(contrib)})
        elif contrib < -0.05:
            protective_factors.append({"feature": f, "value": v, "impact": float(contrib)})
    
    return risk_factors[:3], protective_factors[:3]

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    """Render patient input form"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Generate prediction and report"""
    try:
        data = request.json
        
        # Parse patient data
        patient_data = {
            "age": int(data.get("age", 50)),
            "sex": int(data.get("sex", 1)),
            "cp": int(data.get("cp", 0)),
            "trestbps": int(data.get("trestbps", 120)),
            "chol": int(data.get("chol", 200)),
            "fbs": int(data.get("fbs", 0)),
            "restecg": int(data.get("restecg", 0)),
            "thalach": int(data.get("thalach", 130)),
            "exang": int(data.get("exang", 0)),
            "oldpeak": float(data.get("oldpeak", 0)),
            "slope": int(data.get("slope", 0)),
            "ca": int(data.get("ca", 0)),
            "thal": int(data.get("thal", 2))
        }
        
        name = data.get("name", "Patient")
        
        # Convert to array format and scale
        x = np.array([patient_data[f] for f in FEATURE_NAMES]).reshape(1, -1)
        x_scaled = SCALER.transform(x)
        
        # Get prediction
        prob = MODEL.predict_proba(x_scaled)[0, 1]
        
        # Get risk level info
        risk_info = get_risk_level_info(prob)
        
        # Get SHAP insights
        risk_factors, protective_factors = get_shap_insights(patient_data)
        
        # Format clinical measurements
        clinical_measurements = [
            {"label": "Blood Pressure", "value": f"{patient_data['trestbps']} mmHg"},
            {"label": "Cholesterol", "value": f"{patient_data['chol']} mg/dL"},
            {"label": "Max Heart Rate", "value": f"{patient_data['thalach']} bpm"},
            {"label": "ST Depression", "value": f"{patient_data['oldpeak']}"},
            {"label": "Blood Sugar", "value": "High" if patient_data['fbs'] == 1 else "Normal"},
        ]
        
        # Generate possible diagnoses based on risk score
        possible_diagnosis = []
        if prob > 0.7:
            possible_diagnosis = ["Coronary Artery Disease (CAD)", "Myocardial Infarction", "Unstable Angina"]
        elif prob > 0.5:
            possible_diagnosis = ["Hypertension", "Heart Failure", "Arrhythmia"]
        elif prob > 0.3:
            possible_diagnosis = ["Hyperlipidemia", "Atherosclerosis", "Valve Disease"]
        else:
            possible_diagnosis = ["Normal CVD Status", "Low-risk condition", "Preventive monitoring recommended"]
        
        # Generate suggested diagnostic tests
        suggested_tests = [
            "Ambulatory Blood Pressure Monitoring (ABPM)",
            "Electrocardiogram (ECG)",
            "Echocardiogram",
            "Coronary CT Angiography",
            "Cardiac Stress Test"
        ]
        
        # Extract preliminary questions
        pregnant = "Yes" if data.get("pregnant") == "1" else "No"
        underlying_sickness = "Yes" if data.get("underlying_sickness") == "1" else "No"
        overweight = "Yes" if data.get("overweight") == "1" else "No"
        injury = "Yes" if data.get("injury") == "1" else "No"
        symptoms_duration = data.get("symptoms_duration", "Unknown")
        
        # Prepare response
        report_data = {
            "name": name,
            "age": patient_data["age"],
            "gender": "Male" if patient_data["sex"] == 1 else "Female",
            "height": f"{data.get('height', 170)} cm",
            "weight": f"{data.get('weight', 75)} kg",
            "symptoms": data.get("symptoms", "None reported"),
            "pregnant": pregnant,
            "underlying_sickness": underlying_sickness,
            "overweight": overweight,
            "injury": injury,
            "symptoms_duration": symptoms_duration,
            "risk_percentage": round(prob * 100, 1),
            "risk_level": risk_info["level"],
            "risk_color": risk_info["color"],
            "clinical_measurements": clinical_measurements,
            "possible_diagnosis": possible_diagnosis,
            "suggested_tests": suggested_tests,
            "risk_factors": risk_factors,
            "protective_factors": protective_factors,
            "recommendations": risk_info["recommendations"][:4]
        }
        
        return jsonify({
            "success": True,
            "report": report_data
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

# ─────────────────────────────────────────────
# RUN APP
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Hospital Cardiovascular Assessment System")
    print("="*60)
    print("Starting server at http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)
