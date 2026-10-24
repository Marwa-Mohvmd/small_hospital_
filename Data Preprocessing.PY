# ==================================================
# Hospital Dataset - Data Preprocessing
# No Model Training | No PCA
# For VS Code (.py file)
# ==================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# 1. Load Dataset
file_path = r"C:\LA STORE\Downloads\small_hospital_ml_dataset.xlsx"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"Dataset not found:\n{file_path}\n"
        "Check the file path and name."
    )

df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()

print("=" * 50)
print("ORIGINAL DATASET")
print("=" * 50)

print("Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())


# 2. Create a Copy and Remove Exact Duplicates
df_clean = df.copy()
df_clean = df_clean.drop_duplicates().copy()


# 3. Convert Numeric Columns
numeric_cols = [
    "Age",
    "Billing_Amount_USD",
    "Length_of_Stay_Days",
    "ICU_Admission"
]

for col in numeric_cols:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(
            df_clean[col],
            errors="coerce"
        )


# 4. Check Invalid Values
print("\n" + "=" * 50)
print("INVALID VALUES CHECK")
print("=" * 50)

if "Age" in df_clean.columns:
    print("Age below 0:", (df_clean["Age"] < 0).sum())
    print("Age above 120:", (df_clean["Age"] > 120).sum())

if "Billing_Amount_USD" in df_clean.columns:
    print(
        "Negative Billing:",
        (df_clean["Billing_Amount_USD"] < 0).sum()
    )

if "Length_of_Stay_Days" in df_clean.columns:
    print(
        "Non-positive Length of Stay:",
        (df_clean["Length_of_Stay_Days"] <= 0).sum()
    )


# 5. Validate Target Column
target = "ICU_Admission"

if target not in df_clean.columns:
    raise ValueError(f"Column '{target}' was not found.")

print("\nTarget Values Before Cleaning:")
print(df_clean[target].value_counts(dropna=False))

# Remove rows with missing target
df_clean = df_clean.dropna(subset=[target]).copy()

# Keep only valid binary values
df_clean = df_clean[df_clean[target].isin([0, 1])].copy()
df_clean[target] = df_clean[target].astype(int)

if df_clean[target].nunique() != 2:
    raise ValueError(
        "ICU_Admission must contain both 0 and 1."
    )

if df_clean[target].value_counts().min() < 2:
    raise ValueError(
        "Each target class needs at least 2 rows."
    )


# 6. Separate Features and Target
# Patient_ID is excluded from features
X = df_clean.drop(
    columns=["Patient_ID", target],
    errors="ignore"
)

y = df_clean[target]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


# 7. Identify Numeric and Categorical Features
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool", "string"]
).columns.tolist()

print("\nNumeric Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# 8. Split Data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# 9. Numeric Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


# 10. Categorical Preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# 11. Combine Preprocessing
transformers = []

if numeric_features:
    transformers.append(
        ("num", numeric_pipeline, numeric_features)
    )

if categorical_features:
    transformers.append(
        ("cat", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(transformers=transformers)


# 12. Fit on Training Data Only
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data without fitting again
X_test_processed = preprocessor.transform(X_test)


# 13. Get Processed Feature Names
feature_names = preprocessor.get_feature_names_out()


# 14. Convert Output to Dense DataFrames
def to_dense_array(data):
    if hasattr(data, "toarray"):
        return data.toarray()
    return np.asarray(data)


X_train_processed_df = pd.DataFrame(
    to_dense_array(X_train_processed),
    columns=feature_names,
    index=X_train.index
)

X_test_processed_df = pd.DataFrame(
    to_dense_array(X_test_processed),
    columns=feature_names,
    index=X_test.index
)


# 15. Display Final Results
print("\n" + "=" * 50)
print("PREPROCESSING COMPLETED")
print("=" * 50)

print("Original Dataset Shape:", df.shape)
print("Cleaned Dataset Shape:", df_clean.shape)

print("\nProcessed Training Shape:",
      X_train_processed_df.shape)

print("Processed Testing Shape:",
      X_test_processed_df.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

print("\nFirst 5 Processed Rows:")
print(X_train_processed_df.head())

print("\nMissing Values in Processed Training Data:",
      X_train_processed_df.isnull().sum().sum())

print("Missing Values in Processed Testing Data:",
      X_test_processed_df.isnull().sum().sum())


# 16. Save Processed Data
output_dir = r"C\Users\LA STORE\Downloads\small_hospital_\processed_data"
os.makedirs(output_dir, exist_ok=True)

X_train_processed_df.to_csv(
    os.path.join(output_dir, "X_train.csv"),
    index=False
)

X_test_processed_df.to_csv(
    os.path.join(output_dir, "X_test.csv"),
    index=False
)

y_train.to_csv(
    os.path.join(output_dir, "y_train.csv"),
    index=False
)

y_test.to_csv(
    os.path.join(output_dir, "y_test.csv"),
    index=False
)


# 17. Save the Fitted Preprocessor
joblib.dump(
    preprocessor,
    os.path.join(output_dir, "preprocessor.joblib")
)


# 18. Save the Cleaned Dataset
df_clean.to_csv(
    os.path.join(output_dir, "hospital_cleaned.csv"),
    index=False
)

print("\nAll files saved successfully in:")
print(output_dir)