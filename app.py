
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "customer_churn_model.cbm"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"


@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    return model


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r") as file:
        return json.load(file)


st.title("📊 Customer Churn Prediction")



if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error(
        "Model files are missing. Please check the models/ folder."
    )
    st.stop()


model = load_model()
metadata = load_metadata()

features = metadata["features"]
categorical_features = metadata["categorical_features"]
threshold = float(metadata.get("threshold", 0.5))

st.subheader("Enter Customer Details")

st.info(
    "Provide the customer information below and click "
    "'Predict Churn' to generate a prediction."
)

with st.form("customer_form"):

    customer_data = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):

        container = col1 if i % 2 == 0 else col2

        with container:

            if feature in categorical_features:

                # Common categories in this dataset.
                # Update these if your training dataset uses
                # different category names.
                category_options = {
                    "Gender": ["Male", "Female"],
                    "Subscription Type": [
                        "Basic", "Standard", "Premium"
                    ],
                }

                options = category_options.get(
                    feature, ["Unknown"]
                )

                customer_data[feature] = st.selectbox(
                    feature,
                    options=options,
                    key=f"input_{feature}"
                )

            else:

                numeric_settings = {
                    "Age": (18, 100, 35),
                    "Tenure": (0, 100, 12),
                    "Usage Frequency": (0, 100, 20),
                    "Support Calls": (0, 50, 2),
                    "Payment Delay": (0, 100, 5),
                    "Total Spend": (0.0, 100000.0, 500.0),
                    "Last Interaction": (0, 365, 10),
                }

                minimum, maximum, default = numeric_settings.get(
                    feature, (0.0, 100000.0, 0.0)
                )

                customer_data[feature] = st.number_input(
                    feature,
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    key=f"input_{feature}"
                )

    submitted = st.form_submit_button(
        "🔍 Predict Churn",
        use_container_width=True
    )


if submitted:

    input_df = pd.DataFrame(
        [[customer_data[feature] for feature in features]],
        columns=features
    )

    try:
        probabilities = model.predict_proba(input_df)[0]

        # Find the probability corresponding to churn class 1
        class_labels = list(model.classes_)
        churn_index = class_labels.index(1)
        churn_probability = float(probabilities[churn_index])

        prediction = int(
            churn_probability >= threshold
        )

        st.divider()
        st.subheader("Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Churn Probability",
                f"{churn_probability:.1%}"
            )

        with result_col2:
            if prediction == 1:
                st.error("⚠️ High Churn Risk")
                st.write(
                    "The model classifies this customer as "
                    "likely to churn at the configured threshold."
                )
            else:
                st.success("✅ Lower Churn Risk")
                st.write(
                    "The model classifies this customer as "
                    "less likely to churn at the configured threshold."
                )

        st.progress(churn_probability)

        st.caption(
            f"Decision threshold: {threshold:.2f}. "
            "This is a model prediction, not a guarantee "
            "of the customer's future behavior."
        )

        with st.expander("View submitted customer details"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as error:
        st.error(
            "Prediction failed. Check the input categories "
            "and model feature configuration."
        )
        st.exception(error)
