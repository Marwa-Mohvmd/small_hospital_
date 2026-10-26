import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
X_train = pd.read_csv(os.path.join(base, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(base, "X_test.csv"))
y_train = pd.read_csv(os.path.join(base, "y_train.csv")).values.ravel()
y_test  = pd.read_csv(os.path.join(base, "y_test.csv")).values.ravel()
df = pd.read_csv(os.path.join(base, "hospital_cleaned.csv"))

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    eval_metric="logloss",
    random_state=42
)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
print("accuracy:", accuracy_score(y_test, y_pred))
print("precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

out_dir = os.path.dirname(os.path.abspath(__file__))

# Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="viridis")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"), dpi=150)
plt.show()

# Distribution chart
plt.figure(figsize=(6, 5))
sns.histplot(df["Length_of_Stay_Days"], bins=20, kde=True, color="#0E8C84")
plt.title("Length of Stay Distribution")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "length_of_stay_distribution.png"), dpi=150)
plt.show()