# Customer Churn Prediction

This repository contains a machine learning project for predicting whether a customer is likely to churn (leave a service or subscription) based on customer attributes and behavior.

## Project Overview

The project uses a customer churn dataset and trains a CatBoost classification model to detect customers at high risk of churn. A Streamlit app is included so users can input customer details and get a prediction.

## Repository Structure

- `Dataset/` - Folder containing the churn dataset used for training
- `models/` - Saved trained model and metadata
- `app.py` - Streamlit web application for prediction
- `model.py` - Training script for preprocessing, model fitting, evaluation, and saving the model
- `requirements.txt` - Required Python packages
- `customer_churn_predictions.csv` - Example output predictions file

## Objective

The goal of this project is to build a predictive model that identifies churn risk early so businesses can take preventive action for vulnerable customers.

## Dataset

The dataset contains customer-related information such as:

- Age
- Tenure
- Usage Frequency
- Support Calls
- Payment Delay
- Total Spend
- Last Interaction
- Subscription Type
- Gender

The target variable is:

- `Churn`

## Model

The project uses a CatBoost classifier because it handles categorical features efficiently and produces strong classification performance.

## Tech Stack

- Python
- Pandas
- Scikit-learn
- CatBoost
- Streamlit

## Installation

1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

This starts the Streamlit interface where you can enter customer information and see the churn risk prediction.

## Train the Model

```bash
python model.py
```

This script loads the dataset, processes it, trains the CatBoost model, evaluates the results, and saves the model files in the `models/` folder.

## Prediction Output

The app displays:

- churn probability
- whether the customer is at high or low risk
- a progress bar based on the probability score

## Notes

This project is intended for learning and demonstration of customer churn prediction in a business analytics context.
