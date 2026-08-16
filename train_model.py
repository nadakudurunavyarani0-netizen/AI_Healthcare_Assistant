import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Load dataset
data = pd.read_csv("data/symptoms.csv")

print("Dataset loaded successfully!")
print("Number of records:", len(data))

# 2. Separate input and output
X = data.drop("disease", axis=1)
y = data["disease"]

print("Number of symptoms:", X.shape[1])
print("Diseases:", y.unique())

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.35,
    
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 4. Create improved Random Forest model
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)

# 5. Train model
model.fit(X_train, y_train)

print("Model training completed!")

# 6. Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)
recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)
f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

# 7. Display results
print("\n========== MODEL PERFORMANCE ==========")
print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1-Score :", round(f1 * 100, 2), "%")
print("=======================================\n")

# 8. Create model folder
os.makedirs("model", exist_ok=True)

# 9. Save trained model
joblib.dump(model, "model/disease_model.pkl")

print("Model saved successfully!")
print("File: model/disease_model.pkl")