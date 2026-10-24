import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
X_train = pd.read_csv(os.path.join(base, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(base, "X_test.csv"))
y_train = pd.read_csv(os.path.join(base, "y_train.csv")).values.ravel()
y_test  = pd.read_csv(os.path.join(base, "y_test.csv")).values.ravel()

model = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                               max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy :", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 3))
print("Recall   :", round(recall_score(y_test, y_pred, zero_division=0), 3))
print("F1       :", round(f1_score(y_test, y_pred, zero_division=0), 3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n", classification_report(y_test, y_pred, zero_division=0))


imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("Top 5 important features:\n", imp.head(5).round(3))