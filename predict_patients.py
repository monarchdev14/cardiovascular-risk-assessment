# -*- coding: utf-8 -*-
"""
🏥 HEART DISEASE PREDICTOR - CLEAN INFERENCE
✨ Predict disease risk for patients with clear, visual results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Load & Train Model (One Time)
# ──────────────────────────────────────────────────────────────

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

def setup_model():
    """Train model on full dataset"""
    df = pd.read_csv(UCI_URL, header=None, names=COLUMNS, na_values="?")
    df.dropna(inplace=True)
    df["target"] = (df["target"] > 0).astype(int)
    
    X = df.drop(columns=["target"]).values.astype(np.float64)
    y = df["target"].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, COLUMNS[:-1]

# ──────────────────────────────────────────────────────────────
# Sample Patients (3 Real-World Examples)
# ──────────────────────────────────────────────────────────────

SAMPLE_PATIENTS = [
    {
        "name": "John Smith\n52 years old, Male",
        "age": 52, "sex": 1, "cp": 0, "trestbps": 130, "chol": 220,
        "fbs": 0, "restecg": 1, "thalach": 155, "exang": 0,
        "oldpeak": 1.2, "slope": 2, "ca": 0, "thal": 3
    },
    {
        "name": "Maria Garcia\n45 years old, Female",
        "age": 45, "sex": 0, "cp": 1, "trestbps": 125, "chol": 200,
        "fbs": 1, "restecg": 0, "thalach": 140, "exang": 0,
        "oldpeak": 0.8, "slope": 1, "ca": 0, "thal": 2
    },
    {
        "name": "Robert Johnson\n68 years old, Male",
        "age": 68, "sex": 1, "cp": 2, "trestbps": 145, "chol": 280,
        "fbs": 1, "restecg": 2, "thalach": 120, "exang": 1,
        "oldpeak": 3.5, "slope": 2, "ca": 2, "thal": 3
    },
    {
        "name": "Sarah Wilson\n38 years old, Female",
        "age": 38, "sex": 0, "cp": 0, "trestbps": 118, "chol": 180,
        "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0,
        "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 1
    }
]

# ──────────────────────────────────────────────────────────────
# Make Predictions
# ──────────────────────────────────────────────────────────────

def predict_patient(model, scaler, patient, feature_names):
    """Get prediction for one patient"""
    x = np.array([patient[f] for f in feature_names]).reshape(1, -1)
    x_scaled = scaler.transform(x)
    
    prob = model.predict_proba(x_scaled)[0, 1]
    prediction = "Disease" if prob > 0.5 else "No Disease"
    
    # Risk level
    if prob > 0.7:
        risk = "CRITICAL"
        color = "#d32f2f"  # Red
    elif prob > 0.5:
        risk = "HIGH"
        color = "#f57c00"  # Orange
    elif prob > 0.3:
        risk = "MEDIUM"
        color = "#fbc02d"  # Yellow
    else:
        risk = "LOW"
        color = "#388e3c"  # Green
    
    return prob, prediction, risk, color

# ──────────────────────────────────────────────────────────────
# Create Clean Visualizations
# ──────────────────────────────────────────────────────────────

def create_results_chart(model, scaler, feature_names):
    """Create clean results visualization for all patients"""
    
    results = []
    for patient in SAMPLE_PATIENTS:
        prob, pred, risk, color = predict_patient(model, scaler, patient, feature_names)
        results.append({
            "name": patient["name"],
            "prob": prob * 100,
            "risk": risk,
            "color": color
        })
    
    # Create figure with clean design
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Heart Disease Risk Assessment - Patient Report", 
                 fontsize=18, fontweight="bold", y=0.98)
    axes = axes.flatten()
    
    for idx, (ax, result) in enumerate(zip(axes, results)):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")
        
        # Patient name
        ax.text(50, 85, result["name"], ha="center", va="top", 
               fontsize=13, fontweight="bold")
        
        # Risk indicator circle
        circle = plt.Circle((30, 50), 15, color=result["color"], alpha=0.8)
        ax.add_patch(circle)
        ax.text(30, 50, f"{result['prob']:.0f}%", ha="center", va="center",
               fontsize=16, fontweight="bold", color="white")
        
        # Risk level text
        risk_text = f"Risk: {result['risk']}"
        ax.text(30, 28, risk_text, ha="center", va="center",
               fontsize=11, fontweight="bold", bbox=dict(boxstyle="round", 
               facecolor=result["color"], alpha=0.3))
        
        # Prediction
        pred_text = "LIKELY HAS DISEASE" if result["prob"] > 50 else "LIKELY HEALTHY"
        ax.text(70, 60, pred_text, ha="left", va="center",
               fontsize=11, fontweight="bold", color=result["color"])
        
        # Status box
        status = "⚠️ REQUIRES ATTENTION" if result["prob"] > 50 else "✓ NORMAL"
        ax.text(70, 40, status, ha="left", va="center", fontsize=10)
    
    plt.tight_layout()
    plt.savefig("patient_predictions.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: patient_predictions.png")
    plt.show()

def create_risk_comparison_chart(model, scaler, feature_names):
    """Create bar chart comparing risk levels"""
    
    names = []
    risks = []
    colors_list = []
    
    for patient in SAMPLE_PATIENTS:
        prob, _, _, color = predict_patient(model, scaler, patient, feature_names)
        names.append(patient["name"].split("\n")[0])  # Just first name
        risks.append(prob * 100)
        colors_list.append(color)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(names, risks, color=colors_list, edgecolor="black", linewidth=1.5)
    
    ax.set_xlabel("Disease Probability (%)", fontsize=12, fontweight="bold")
    ax.set_title("Heart Disease Risk Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlim(0, 100)
    
    # Add value labels
    for i, (bar, risk) in enumerate(zip(bars, risks)):
        ax.text(risk + 2, i, f"{risk:.1f}%", va="center", fontweight="bold")
        
        # Add risk level label
        if risk > 70:
            label = "CRITICAL"
        elif risk > 50:
            label = "HIGH"
        elif risk > 30:
            label = "MEDIUM"
        else:
            label = "LOW"
        ax.text(5, i, label, va="center", fontweight="bold", color="white", fontsize=10)
    
    # Add reference lines
    ax.axvline(50, color="red", linestyle="--", linewidth=2, alpha=0.5, label="Disease Threshold")
    ax.axvline(30, color="orange", linestyle="--", linewidth=2, alpha=0.5, label="Medium Risk")
    ax.legend(loc="lower right")
    
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("risk_comparison.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: risk_comparison.png")
    plt.show()

def print_results(model, scaler, feature_names):
    """Print clean text results"""
    
    print("\n" + "="*70)
    print(" "*15 + "HEART DISEASE RISK PREDICTION REPORT")
    print("="*70)
    
    for i, patient in enumerate(SAMPLE_PATIENTS, 1):
        prob, pred, risk, _ = predict_patient(model, scaler, patient, feature_names)
        
        print(f"\n📋 PATIENT {i}: {patient['name'].replace(chr(10), ' | ')}")
        print(f"   Risk Probability: {prob*100:.1f}%")
        print(f"   Risk Level:       {risk}")
        print(f"   Prediction:       {pred}")
        
        if prob > 0.7:
            print(f"   ⚠️  ACTION:        URGENT - Schedule immediate cardiac consultation")
        elif prob > 0.5:
            print(f"   ⚠️  ACTION:        Schedule doctor appointment soon")
        else:
            print(f"   ✓  ACTION:        Continue routine check-ups")
    
    print("\n" + "="*70)

# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Loading and training model...")
    model, scaler, feature_names = setup_model()
    print("✓ Model ready!\n")
    
    print_results(model, scaler, feature_names)
    
    print("\n📊 Generating visualizations...")
    create_results_chart(model, scaler, feature_names)
    create_risk_comparison_chart(model, scaler, feature_names)
    
    print("\n✨ All done! Check patient_predictions.png and risk_comparison.png")
