from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import joblib
import re

app = Flask(__name__)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "disease_model.pkl"
)

SYMPTOMS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "symptoms.csv"
)

MEDICATIONS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "medications.csv"
)

SIDE_EFFECTS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "side_effects.csv"
)

DRUG_INTERACTIONS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "drug_interactions.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

# ============================================================
# LOAD DISEASE MODEL
# ============================================================

model = None

try:

    if os.path.exists(MODEL_PATH):

        model = joblib.load(MODEL_PATH)

        print("==============================================")
        print("Disease model loaded successfully!")
        print("Model:", type(model).__name__)
        print("==============================================")

    else:

        print("ERROR: disease_model.pkl not found!")
        print("Expected location:")
        print(MODEL_PATH)

except Exception as e:

    print("==============================================")
    print("ERROR loading disease model")
    print("==============================================")
    print(str(e))
# ============================================================
# LOAD DATASETS
# ============================================================

def load_csv(path):

    try:

        if os.path.exists(path):

            df = pd.read_csv(path)

            # Remove accidental spaces from column names
            df.columns = [
                str(col).strip()
                for col in df.columns
            ]

            return df

        print("Dataset not found:", path)

        return pd.DataFrame()

    except Exception as e:

        print("Error loading:", path)
        print(e)

        return pd.DataFrame()


symptoms_df = load_csv(SYMPTOMS_PATH)

medications_df = load_csv(MEDICATIONS_PATH)

side_effects_df = load_csv(SIDE_EFFECTS_PATH)

drug_interactions_df = load_csv(DRUG_INTERACTIONS_PATH)


# ============================================================
# SYMPTOMS USED BY YOUR PROJECT
# ============================================================

DEFAULT_SYMPTOMS = [

    "fever",
    "cough",
    "fatigue",
    "headache",
    "body_pain",
    "sore_throat",
    "runny_nose",
    "nausea",
    "vomiting",
    "diarrhea"

]


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    value = str(value).strip().lower()

    value = value.replace("-", "_")
    value = value.replace(" ", "_")

    value = re.sub(
        r"[^a-z0-9_]",
        "",
        value
    )

    return value


# ============================================================
# FIND MODEL FEATURE NAMES
# ============================================================

def get_model_features():

    # Most sklearn models store feature names here
    if model is not None:

        if hasattr(model, "feature_names_in_"):

            return [
                str(x)
                for x in model.feature_names_in_
            ]

    # Fallback to project symptoms
    return DEFAULT_SYMPTOMS.copy()


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

def create_input_dataframe(selected_symptoms):

    selected_symptoms = [
        clean_text(x)
        for x in selected_symptoms
    ]

    selected_symptoms = set(
        selected_symptoms
    )

    model_features = get_model_features()

    input_data = {}

    for feature in model_features:

        clean_feature = clean_text(feature)

        if clean_feature in selected_symptoms:

            input_data[feature] = 1

        else:

            input_data[feature] = 0

    input_df = pd.DataFrame(
        [input_data],
        columns=model_features
    )

    return input_df


# ============================================================
# DISEASE PREDICTION
# ============================================================

def predict_disease(selected_symptoms):

    if model is None:

        return (
            "Model Not Available",
            0.0
        )

    try:

        input_df = create_input_dataframe(
            selected_symptoms
        )

        print("\nInput given to model:")
        print(input_df)

        # -----------------------------------------
        # Prediction
        # -----------------------------------------

        prediction = model.predict(
            input_df
        )[0]

        prediction = str(prediction)

        # -----------------------------------------
        # Confidence
        # -----------------------------------------

        confidence = 0.0

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_df
            )[0]

            confidence = float(
                np.max(probabilities) * 100
            )

        return (
            prediction,
            round(confidence, 2)
        )

    except Exception as e:

        print("\nPrediction Error:")
        print(e)

        return (
            "Prediction Error",
            0.0
        )


# ============================================================
# FIND MEDICATION INFORMATION
# ============================================================

def get_medication_info(disease):

    if medications_df.empty:

        return (
            "Not available in dataset",
            "Consult a healthcare professional"
        )

    try:

        columns = [
            clean_text(c)
            for c in medications_df.columns
        ]

        medications_df.columns = columns

        disease_column = None
        medicine_column = None
        alternative_column = None

        # Find disease column
        for col in medications_df.columns:

            if col in [
                "disease",
                "condition",
                "diagnosis"
            ]:

                disease_column = col
                break

        # Find medicine column
        for col in medications_df.columns:

            if col in [
                "medicine",
                "medication",
                "drug",
                "recommended_drug"
            ]:

                medicine_column = col
                break

        # Find alternative column
        for col in medications_df.columns:

            if col in [
                "alternative",
                "alternative_drug",
                "alternative_medicine"
            ]:

                alternative_column = col
                break

        if disease_column is None:

            return (
                "Not available",
                "Consult a healthcare professional"
            )

        disease_clean = clean_text(disease)

        matches = medications_df[
            medications_df[disease_column]
            .astype(str)
            .apply(clean_text)
            == disease_clean
        ]

        if matches.empty:

            return (
                "No medication information found",
                "Consult a healthcare professional"
            )

        row = matches.iloc[0]

        if medicine_column:

            medication = str(
                row[medicine_column]
            )

        else:

            medication = "Not available"

        if alternative_column:

            alternative = str(
                row[alternative_column]
            )

        else:

            alternative = (
                "Consult a healthcare professional"
            )

        return (
            medication,
            alternative
        )

    except Exception as e:

        print("Medication lookup error:", e)

        return (
            "Not available",
            "Consult a healthcare professional"
        )


# ============================================================
# FIND SIDE EFFECT INFORMATION
# ============================================================

def get_side_effect_info(medication):

    if side_effects_df.empty:

        return (
            "No side-effect information found.",
            "Unknown"
        )

    try:

        df = side_effects_df.copy()

        df.columns = [
            clean_text(c)
            for c in df.columns
        ]

        medicine_column = None
        side_effect_column = None
        severity_column = None

        # Medicine column
        for col in df.columns:

            if col in [
                "medicine",
                "medication",
                "drug"
            ]:

                medicine_column = col
                break

        # Side effect column
        for col in df.columns:

            if col in [
                "side_effect",
                "side_effects",
                "effects",
                "common_side_effect"
            ]:

                side_effect_column = col
                break

        # Severity column
        for col in df.columns:

            if col in [
                "severity",
                "risk",
                "risk_level"
            ]:

                severity_column = col
                break

        if medicine_column is None:

            return (
                "No side-effect information found.",
                "Unknown"
            )

        medication_clean = clean_text(
            medication
        )

        matches = df[
            df[medicine_column]
            .astype(str)
            .apply(clean_text)
            == medication_clean
        ]

        if matches.empty:

            return (
                "No side-effect information found.",
                "Unknown"
            )

        row = matches.iloc[0]

        if side_effect_column:

            effects = str(
                row[side_effect_column]
            )

        else:

            effects = (
                "Side-effect information unavailable."
            )

        if severity_column:

            severity = str(
                row[severity_column]
            )

        else:

            severity = "Unknown"

        return (
            effects,
            severity
        )

    except Exception as e:

        print("Side-effect lookup error:", e)

        return (
            "No side-effect information found.",
            "Unknown"
        )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    symptoms = DEFAULT_SYMPTOMS.copy()

    # If dataset contains symptom columns,
    # use them when possible.

    if not symptoms_df.empty:

        possible_symptoms = []

        for column in symptoms_df.columns:

            clean_column = clean_text(column)

            if clean_column not in [
                "disease",
                "condition",
                "diagnosis",
                "prognosis"
            ]:

                possible_symptoms.append(
                    clean_column
                )

        if possible_symptoms:

            symptoms = possible_symptoms

    return render_template(
        "index.html",
        symptoms=symptoms
    )


# ============================================================
# DISEASE PREDICTION ROUTE
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # Get selected symptoms
        selected_symptoms = request.form.getlist(
            "symptoms"
        )

        print("\nSelected symptoms:")
        print(selected_symptoms)

        # -----------------------------------------
        # No symptoms selected
        # -----------------------------------------

        if not selected_symptoms:

            return render_template(
                "result.html",

                prediction="No symptoms selected",

                confidence=0,

                medication="Not available",

                alternative="Please select symptoms",

                side_effects="Not available",

                severity="Unknown"
            )

        # -----------------------------------------
        # Disease prediction
        # -----------------------------------------

        prediction, confidence = predict_disease(
            selected_symptoms
        )

        # -----------------------------------------
        # Medication information
        # -----------------------------------------

        medication, alternative = (
            get_medication_info(prediction)
        )

        # -----------------------------------------
        # Side-effect monitoring
        # -----------------------------------------

        side_effects, severity = (
            get_side_effect_info(medication)
        )

        # -----------------------------------------
        # Send everything to result.html
        # -----------------------------------------

        return render_template(

            "result.html",

            prediction=prediction,

            confidence=confidence,

            medication=medication,

            alternative=alternative,

            side_effects=side_effects,

            severity=severity
        )

    except Exception as e:

        print("\nPREDICTION ROUTE ERROR:")
        print(e)

        return render_template(

            "result.html",

            prediction="Unable to make prediction",

            confidence=0,

            medication="Not available",

            alternative="Consult a healthcare professional",

            side_effects="Not available",

            severity="Unknown"
        )


# ============================================================
# DRUG INTERACTION PAGE
# ============================================================

@app.route(
    "/drug-interaction",
    methods=["GET", "POST"]
)
def drug_interaction():

    result = None

    if request.method == "POST":

        drug1 = request.form.get(
            "drug1",
            ""
        ).strip()

        drug2 = request.form.get(
            "drug2",
            ""
        ).strip()

        result = check_drug_interaction(
            drug1,
            drug2
        )

        return render_template(
            "drug_interaction.html",
            result=result,
            drug1=drug1,
            drug2=drug2
        )

    return render_template(
        "drug_interaction.html",
        result=None,
        drug1="",
        drug2=""
    )


# ============================================================
# DRUG INTERACTION CHECKER
# ============================================================

def check_drug_interaction(
    drug1,
    drug2
):

    if drug_interactions_df.empty:

        return {

            "found": False,

            "drug1": drug1,

            "drug2": drug2,

            "interaction":
                "Reference interaction dataset not available.",

            "risk":
                "Unknown"

        }

    try:

        df = drug_interactions_df.copy()

        df.columns = [
            clean_text(c)
            for c in df.columns
        ]

        print("\nDrug interaction columns:")
        print(df.columns.tolist())

        # -----------------------------------------
        # Identify columns
        # -----------------------------------------

        drug1_column = None
        drug2_column = None
        interaction_column = None
        risk_column = None

        for col in df.columns:

            if col in [
                "drug1",
                "medicine1",
                "medication1"
            ]:

                drug1_column = col

            elif col in [
                "drug2",
                "medicine2",
                "medication2"
            ]:

                drug2_column = col

            elif col in [
                "interaction",
                "interaction_text",
                "description"
            ]:

                interaction_column = col

            elif col in [
                "risk",
                "risk_level",
                "severity"
            ]:

                risk_column = col

        if (
            drug1_column is None
            or drug2_column is None
        ):

            return {

                "found": False,

                "drug1": drug1,

                "drug2": drug2,

                "interaction":
                    "Invalid interaction dataset format.",

                "risk":
                    "Unknown"

            }

        drug1_clean = clean_text(drug1)
        drug2_clean = clean_text(drug2)

        df["_drug1_clean"] = (
            df[drug1_column]
            .astype(str)
            .apply(clean_text)
        )

        df["_drug2_clean"] = (
            df[drug2_column]
            .astype(str)
            .apply(clean_text)
        )

        # -----------------------------------------
        # Check both directions
        # -----------------------------------------

        match = df[
            (
                (
                    df["_drug1_clean"]
                    == drug1_clean
                )
                &
                (
                    df["_drug2_clean"]
                    == drug2_clean
                )
            )
            |
            (
                (
                    df["_drug1_clean"]
                    == drug2_clean
                )
                &
                (
                    df["_drug2_clean"]
                    == drug1_clean
                )
            )
        ]

        # -----------------------------------------
        # Interaction found
        # -----------------------------------------

        if not match.empty:

            row = match.iloc[0]

            if interaction_column:

                interaction_text = str(
                    row[interaction_column]
                )

            else:

                interaction_text = (
                    "Interaction found in reference dataset."
                )

            if risk_column:

                risk = str(
                    row[risk_column]
                )

            else:

                risk = "Unknown"

            return {

                "found": True,

                "drug1": drug1,

                "drug2": drug2,

                "interaction":
                    interaction_text,

                "risk":
                    risk

            }

        # -----------------------------------------
        # No interaction found
        # -----------------------------------------

        return {

            "found": False,

            "drug1": drug1,

            "drug2": drug2,

            "interaction":
                "No interaction found in the reference dataset.",

            "risk":
                "Unknown"

        }

    except Exception as e:

        print(
            "Drug interaction error:",
            e
        )

        return {

            "found": False,

            "drug1": drug1,

            "drug2": drug2,

            "interaction":
                "Unable to check the interaction.",

            "risk":
                "Unknown"

        }


# ============================================================
# ERROR HANDLER
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return """

    <h2>Page Not Found</h2>

    <p>
        The requested page does not exist.
    </p>

    <a href="/">
        Back to Healthcare Assistant
    </a>

    """, 404


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "AI-POWERED SMART HEALTHCARE ASSISTANT"
    )

    print("=" * 60)

    print(
        "Disease Prediction: ENABLED"
    )

    print(
        "Drug Recommendation: ENABLED"
    )

    print(
        "Side-Effect Monitoring: ENABLED"
    )

    print(
        "Drug Interaction Checker: ENABLED"
    )

    print("=" * 60)

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )