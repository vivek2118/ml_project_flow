import pandas as pd
import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv('iris.csv')
X = data.data
y = data.target
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
# Prediction
pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, pred)


# Track experiment
with mlflow.start_run():
    mlflow.log_metric("accuracy", accuracy)


# Save model
joblib.dump(model, "model.pkl")
print("Model trained successfully!")
print("Accuracy:", accuracy)