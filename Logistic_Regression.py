import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
X_train = pd.read_csv(os.path.join(base, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(base, "X_test.csv"))
y_train = pd.read_csv(os.path.join(base, "y_train.csv")).values.ravel()
y_test  = pd.read_csv(os.path.join(base, "y_test.csv")).values.ravel()

num_cols = [c for c in X_train.columns if c.startswith("num__")]
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy :", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred), 3))
print("Recall   :", round(recall_score(y_test, y_pred), 3))
print("F1       :", round(f1_score(y_test, y_pred), 3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n", classification_report(y_test, y_pred))