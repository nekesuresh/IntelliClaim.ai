import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import time

# Load datasets
print("Loading datasets...")
train_df = pd.read_csv("training_data.csv")
test_df = pd.read_csv("testing_data.csv")

# Separate features and target
X_train = train_df.drop(columns=["Category"])
y_train = train_df["Category"]
X_test = test_df

# Define columns
categorical_cols = ["land_surface_condition", "foundation_type", "roof_type", "ground_floor_type", 
                    "other_floor_type", "position", "plan_configuration", "legal_ownership_status"]
numerical_cols = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id", "count_floors_pre_eq", 
                  "age", "area_percentage", "height_percentage", "count_families"]
binary_cols = [col for col in X_train.columns if col.startswith("has_")]

# Feature engineering for both training and testing data
print("Creating interaction, polynomial, and cluster-based features...")
for df in [X_train, X_test]:
    df["age_height_interaction"] = df["age"] * df["height_percentage"]
    df["floors_families_interaction"] = df["count_floors_pre_eq"] * df["count_families"]
    df["age_squared"] = df["age"] ** 2
    df["height_percentage_squared"] = df["height_percentage"] ** 2

# Cluster-based features
kmeans = KMeans(n_clusters=10, random_state=42)
X_train["cluster"] = kmeans.fit_predict(X_train[numerical_cols])
X_test["cluster"] = kmeans.predict(X_test[numerical_cols])

# Add new features to numerical columns
numerical_cols.extend(["age_height_interaction", "floors_families_interaction", "age_squared", "height_percentage_squared", "cluster"])

# Preprocessing
print("Setting up preprocessing pipeline...")
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numerical_cols + binary_cols)
    ]
)

# Models
print("Initializing models...")
lgb_model = LGBMClassifier(
    random_state=42,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_child_samples=20,
    class_weight='balanced',
    objective='binary',
    metric='auc',
    verbose=1
)

xgb_model = XGBClassifier(
    random_state=42,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
    objective='binary:logistic',
    eval_metric='auc',
    verbosity=1
)

cat_model = CatBoostClassifier(
    random_state=42,
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    auto_class_weights='Balanced',
    verbose=100
)

rf_model = RandomForestClassifier(
    random_state=42,
    n_estimators=500,
    max_depth=6,
    class_weight='balanced',
    verbose=1
)

# Stacking ensemble with LightGBM as meta-model
print("Setting up stacking ensemble...")
stacking_model = StackingClassifier(
    estimators=[
        ('lgb', lgb_model),
        ('xgb', xgb_model),
        ('cat', cat_model),
        ('rf', rf_model)
    ],
    final_estimator=LGBMClassifier(random_state=42, verbose=-1),  # Non-linear meta-model
    cv=5
)

# Pipeline
print("Creating pipeline...")
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('adasyn', ADASYN(random_state=42)),  # Use ADASYN for oversampling
    ('model', stacking_model)
])

# Training
print("Training the model...")
start_time = time.time()
pipeline.fit(X_train, y_train)
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

# Evaluation
print("Evaluating on training data...")
y_train_pred = pipeline.predict(X_train)
y_train_pred_proba = pipeline.predict_proba(X_train)[:, 1]

print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Training ROC-AUC:", roc_auc_score(y_train, y_train_pred_proba))
print("\nTraining Classification Report:\n", classification_report(y_train, y_train_pred))

# Cross-validation
print("Running cross-validation...")
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
print("Cross-Validation ROC-AUC Scores:", cv_scores)
print("Mean ROC-AUC:", cv_scores.mean())

# Predictions
print("Predicting on test data...")
y_test_pred = pipeline.predict(X_test)

# Save predictions
submission_df = pd.DataFrame({
    "building_id": X_test.index + 1,
    "Category": y_test_pred
})
submission_df.to_csv("submission_predictions_ensemble.csv", index=False)
print("\nPredictions saved to 'submission_predictions_ensemble.csv'")

# Predictions with fraud probability scores
print("Predicting fraud probabilities on test data...")
y_test_prob = pipeline.predict_proba(X_test)[:, 1]  # Probability of fraud (class 1)

# Save fraud probability scores
fraud_prob_df = pd.DataFrame({
    "claim_id": X_test.index + 1,  # Adjust as per your data format
    "fraud_probability": y_test_prob
})

fraud_prob_df.to_csv("fraud_probabilities.csv", index=False)
print("\nFraud probabilities saved to 'fraud_probabilities.csv'")
