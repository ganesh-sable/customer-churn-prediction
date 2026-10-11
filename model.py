import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

train = pd.read_csv("Dataset/customer_churn_dataset-training-master.csv")

print(train.head())

print("Dataset shape:", train.shape)

print(train[train.isnull().any(axis=1)])

print("\nMissing values per column:")
print(train.isnull().sum())


print("Before cleaning:", train.shape)

train = train.dropna(how="all").copy()

train.reset_index(drop=True, inplace=True)

print("After cleaning:", train.shape)
print("\nMissing values:")
print(train.isnull().sum().sum())


print(train.info())

print("\nChurn distribution:")
print(train["Churn"].value_counts())

print("\nChurn percentage:")
print(train["Churn"].value_counts(normalize=True) * 100)

print("\nUnique values:")
print(train.nunique())


train = train.drop(columns = "CustomerID")

X = train.drop(columns = ["Churn", "Contract Length"])
y = train["Churn"]

cat_cols = X.select_dtypes(include= "object").columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size= 0.2,
    random_state= 42,
    stratify = y
)

from catboost import CatBoostClassifier
model = CatBoostClassifier(
    iterations= 2000,
    learning_rate= 0.03,
    depth= 6,
    l2_leaf_reg= 3.0,
    loss_function = "Logloss",
    eval_metric= "AUC",
    auto_class_weights= "Balanced",
    random_seed= 42,
    verbose= 100
)

model.fit(X_train, y_train, 
        eval_set=(X_test, y_test), 
        cat_features= cat_cols,
        early_stopping_rounds=100
)

y_pred = model.predict(X_test)
ac = accuracy_score(y_test, y_pred)
print(ac)

print(X_train.shape, X_test.shape)
print(X_train.columns.tolist())
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))


print("Train score:", model.score(X_train, y_train))
print("Test score:", model.score(X_test, y_test))

print(train.groupby("Churn")[
    ["Age", "Tenure", "Usage Frequency",
     "Support Calls", "Payment Delay",
     "Total Spend", "Last Interaction"]
].mean())


from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import pandas as pd

y_pred = model.predict(X_test).ravel()
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

print("\nExact duplicate feature rows:")
print(X_train.duplicated().sum())

import os
import json
import joblib

os.makedirs("models", exist_ok=True)

model.save_model("models/customer_churn_model.cbm")

metadata = {
    "features": X_train.columns.tolist(),
    "categorical_features": cat_cols,
    "target": "Churn",
    "threshold": 0.5,
    "model_type": "CatBoostClassifier"
}

with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Model and metadata saved successfully!")

