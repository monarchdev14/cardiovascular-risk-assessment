# API Documentation

## Base URL

```
http://127.0.0.1:5000
```

## Endpoints

### `GET /`

Serves the patient input form web interface.

**Response:** HTML page with the patient data input form and report display.

---

### `POST /api/predict`

Generate a cardiovascular risk assessment report for a patient.

#### Request Headers

| Header | Value |
|--------|-------|
| `Content-Type` | `application/json` |

#### Request Body Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | No | Patient name (default: "Patient") |
| `age` | integer | Yes | Patient age in years |
| `sex` | integer | Yes | 1 = Male, 0 = Female |
| `cp` | integer | Yes | Chest pain type (0-3) |
| `trestbps` | integer | Yes | Resting blood pressure (mm Hg) |
| `chol` | integer | Yes | Serum cholesterol (mg/dL) |
| `fbs` | integer | Yes | Fasting blood sugar > 120 mg/dL (1 = true, 0 = false) |
| `restecg` | integer | Yes | Resting ECG results (0-2) |
| `thalach` | integer | Yes | Maximum heart rate achieved |
| `exang` | integer | Yes | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | float | Yes | ST depression induced by exercise |
| `slope` | integer | Yes | Slope of peak exercise ST segment (0-2) |
| `ca` | integer | Yes | Number of major vessels colored by fluoroscopy (0-3) |
| `thal` | integer | Yes | Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect) |
| `height` | integer | No | Patient height in cm |
| `weight` | integer | No | Patient weight in kg |
| `symptoms` | string | No | Description of symptoms |
| `pregnant` | string | No | "1" for yes, "0" for no |
| `underlying_sickness` | string | No | "1" for yes, "0" for no |
| `overweight` | string | No | "1" for yes, "0" for no |
| `injury` | string | No | "1" for yes, "0" for no |
| `symptoms_duration` | string | No | Duration of symptoms |

#### Success Response (200)

```json
{
  "success": true,
  "report": {
    "name": "John Doe",
    "age": 52,
    "gender": "Male",
    "height": "175 cm",
    "weight": "82 kg",
    "symptoms": "Chest pain during exercise",
    "pregnant": "No",
    "underlying_sickness": "Yes",
    "overweight": "Yes",
    "injury": "No",
    "symptoms_duration": "2 weeks",
    "risk_percentage": 90.1,
    "risk_level": "HIGH RISK",
    "risk_color": "#d62728",
    "clinical_measurements": [
      {"label": "Blood Pressure", "value": "145 mmHg"},
      {"label": "Cholesterol", "value": "280 mg/dL"},
      {"label": "Max Heart Rate", "value": "120 bpm"},
      {"label": "ST Depression", "value": "3.5"},
      {"label": "Blood Sugar", "value": "High"}
    ],
    "possible_diagnosis": [
      "Coronary Artery Disease (CAD)",
      "Myocardial Infarction",
      "Unstable Angina"
    ],
    "suggested_tests": [
      "Ambulatory Blood Pressure Monitoring (ABPM)",
      "Electrocardiogram (ECG)",
      "Echocardiogram",
      "Coronary CT Angiography",
      "Cardiac Stress Test"
    ],
    "risk_factors": [
      {"feature": "thal", "value": 3, "impact": 0.42},
      {"feature": "ca", "value": 2, "impact": 0.31}
    ],
    "protective_factors": [],
    "recommendations": [
      "URGENT: Schedule cardiology consultation within 1 week",
      "Undergo comprehensive cardiac evaluation and stress testing",
      "Discuss medication options with cardiovascular specialist",
      "Begin daily moderate exercise (with doctor approval)"
    ]
  }
}
```

#### Error Response (400)

```json
{
  "success": false,
  "error": "Error message describing what went wrong"
}
```

#### Risk Level Classification

| Risk Percentage | Level | Color | Urgency |
|----------------|-------|-------|---------|
| > 70% | HIGH RISK | 🔴 `#d62728` | Immediate medical consultation |
| > 50% | MODERATE RISK | 🟠 `#ff7f0e` | Schedule appointment within 2-4 weeks |
| > 30% | MODERATE-LOW RISK | 🟡 `#fbc02d` | Annual cardiovascular check-up |
| ≤ 30% | LOW RISK | 🟢 `#2ca02c` | Continue routine check-ups |

---

## cURL Examples

### Basic Prediction

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Patient",
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 260,
    "fbs": 1,
    "restecg": 1,
    "thalach": 130,
    "exang": 1,
    "oldpeak": 2.5,
    "slope": 2,
    "ca": 1,
    "thal": 3
  }'
```

### Python Example

```python
import requests

response = requests.post(
    "http://127.0.0.1:5000/api/predict",
    json={
        "name": "Jane Smith",
        "age": 45,
        "sex": 0,
        "cp": 1,
        "trestbps": 125,
        "chol": 200,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 0.5,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }
)

data = response.json()
print(f"Risk: {data['report']['risk_percentage']}% - {data['report']['risk_level']}")
```
