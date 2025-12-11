import pandas as pd
import joblib
from pathlib import Path

# Resolve artifacts directory relative to this file (cross-platform)
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = BASE_DIR / "artifacts"

# Safe loader that raises a clear error if a file is missing
def _load_joblib(name):
    p = ARTIFACTS / name
    if not p.exists():
        raise FileNotFoundError(f"Missing artifact: {p}")
    return joblib.load(p)

# Load models / scalers
model_young = _load_joblib("model_young.joblib")
model_rest  = _load_joblib("model_rest.joblib")
scaler_young = _load_joblib("scaler_young.joblib")
scaler_rest  = _load_joblib("scaler_rest.joblib")


def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    if medical_history is None:
        medical_history = "none"
    diseases = str(medical_history).lower().split(" & ")
    total_risk_score = sum(risk_scores.get(d.strip(), 0) for d in diseases)
    max_score = 14  # 8 + 6
    min_score = 0
    if max_score == min_score:
        return 0.0
    return (total_risk_score - min_score) / (max_score - min_score)


def preprocess_input(input_dict):
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    for key, value in (input_dict or {}).items():
        if key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1
        elif key == 'Region':
            if value == 'Northwest':
                df['region_Northwest'] = 1
            elif value == 'Southeast':
                df['region_Southeast'] = 1
            elif value == 'Southwest':
                df['region_Southwest'] = 1
        elif key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_Unmarried'] = 1
        elif key == 'BMI Category':
            if value == 'Obesity':
                df['bmi_category_Obesity'] = 1
            elif value == 'Overweight':
                df['bmi_category_Overweight'] = 1
            elif value == 'Underweight':
                df['bmi_category_Underweight'] = 1
        elif key == 'Smoking Status':
            if value == 'Occasional':
                df['smoking_status_Occasional'] = 1
            elif value == 'Regular':
                df['smoking_status_Regular'] = 1
        elif key == 'Employment Status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_Self-Employed'] = 1
        elif key == 'Insurance Plan':
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key == 'Age':
            df['age'] = value
        elif key == 'Number of Dependants':
            df['number_of_dependants'] = value
        elif key == 'Income in Lakhs':
            df['income_lakhs'] = value
        elif key == "Genetical Risk":
            df['genetical_risk'] = value

    # calculate normalized risk safely
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict.get('Medical History', 'none'))

    # ensure numeric columns exist for scaling
    for col in ['age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score']:
        if col not in df.columns:
            df[col] = 0

    df = handle_scaling(int(input_dict.get('Age', 30)), df)
    return df


def handle_scaling(age, df):
    # pick scaler object
    scaler_object = scaler_young if age <= 25 else scaler_rest

    # Expect scaler_object to be either:
    #  - a dict: {'cols_to_scale': [...], 'scaler': sklearn_scaler}
    #  - or a tuple/list like (scaler, cols_to_scale) [backwards compatible handling]
    if isinstance(scaler_object, dict):
        cols_to_scale = scaler_object.get('cols_to_scale', [])
        scaler = scaler_object.get('scaler')
    elif isinstance(scaler_object, (list, tuple)) and len(scaler_object) == 2:
        scaler, cols_to_scale = scaler_object[0], scaler_object[1]
    else:
        raise ValueError("Unsupported scaler_object format; expected dict or (scaler, cols_list).")

    if not cols_to_scale or scaler is None:
        return df

    # ensure columns exist with default values
    for c in cols_to_scale:
        if c not in df.columns:
            df[c] = 0

    # provide placeholder income_level if scaler expects it
    if 'income_level' not in df.columns:
        df['income_level'] = 0

    # perform scaling (will preserve numeric dtype)
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # cleanup placeholder
    if 'income_level' in df.columns:
        df.drop('income_level', axis=1, inplace=True)

    return df


def predict(input_dict):
    input_df = preprocess_input(input_dict)
    age = int(input_dict.get('Age', 30))
    if age <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)
    return int(prediction[0])
