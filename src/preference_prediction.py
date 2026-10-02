import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# FOODLENS - CUSTOMER PREFERENCE PREDICTION
# ============================================================

print("=" * 70)
print("              FOODLENS PREFERENCE PREDICTION")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load customer and order data
# ------------------------------------------------------------

print("\n[1/7] Loading customer and order data...")

customer_file = os.path.join(
    "data",
    "processed",
    "customer_features.csv"
)

orders_file = os.path.join(
    "data",
    "processed",
    "processed_orders.csv"
)

df = pd.read_csv(customer_file)
orders = pd.read_csv(orders_file)

print(f"Customers loaded : {len(df):,}")
print(f"Orders loaded    : {len(orders):,}")


# ------------------------------------------------------------
# 2. Create cuisine history features
# ------------------------------------------------------------

print("\n[2/7] Creating cuisine history features...")

cuisines = sorted(orders["cuisine"].unique())

cuisine_counts = pd.crosstab(
    orders["customer_id"],
    orders["cuisine"]
)

# Make sure every cuisine exists as a column
cuisine_counts = cuisine_counts.reindex(
    columns=cuisines,
    fill_value=0
)

# Add clear feature names
cuisine_counts.columns = [
    f"orders_{cuisine.lower().replace(' ', '_')}"
    for cuisine in cuisine_counts.columns
]

cuisine_counts = cuisine_counts.reset_index()

# Merge cuisine history with customer profiles
df = df.merge(
    cuisine_counts,
    on="customer_id",
    how="left"
)

# Fill customers with no cuisine history
df = df.fillna(0)

print(f"Cuisine history features created: {len(cuisines)}")


# ------------------------------------------------------------
# 3. Select features
# ------------------------------------------------------------

print("\n[3/7] Preparing prediction features...")

behavior_features = [
    "total_orders",
    "average_order_value",
    "total_spending",
    "average_rating",
    "average_delivery_time",
    "average_discount",
    "average_distance",
    "average_quantity",
    "average_delivery_fee"
]

cuisine_features = [
    f"orders_{cuisine.lower().replace(' ', '_')}"
    for cuisine in cuisines
]

features = behavior_features + cuisine_features

target = "favorite_cuisine"

X = df[features]
y = df[target]

print("\nBehavior features:")
for feature in behavior_features:
    print(f"   ✓ {feature}")

print("\nCuisine history features:")
for feature in cuisine_features:
    print(f"   ✓ {feature}")

print(f"\nTotal features: {len(features)}")
print(f"Target: {target}")
print(f"Number of cuisine classes: {y.nunique()}")


# ------------------------------------------------------------
# 4. Train-test split
# ------------------------------------------------------------

print("\n[4/7] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples : {len(X_test):,}")


# ------------------------------------------------------------
# 5. Train Random Forest
# ------------------------------------------------------------

print("\n[5/7] Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("✓ Model training completed")


# ------------------------------------------------------------
# 6. Evaluate model
# ------------------------------------------------------------

print("\n[6/7] Evaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\nPrediction Accuracy: {accuracy:.4f}")
print(f"Prediction Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------

print("\nFeature Importance:")

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print(
    importance.head(15).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 7. Save model and predictions
# ------------------------------------------------------------

print("\n[7/7] Saving model and predictions...")

os.makedirs(
    "models",
    exist_ok=True
)

model_file = os.path.join(
    "models",
    "preference_prediction_model.pkl"
)

joblib.dump(
    model,
    model_file
)

importance_file = os.path.join(
    "data",
    "processed",
    "preference_feature_importance.csv"
)

importance.to_csv(
    importance_file,
    index=False
)


# ------------------------------------------------------------
# Generate predictions
# ------------------------------------------------------------

df["predicted_cuisine"] = model.predict(
    df[features]
)

prediction_file = os.path.join(
    "data",
    "processed",
    "customer_predictions.csv"
)

df.to_csv(
    prediction_file,
    index=False
)


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("       PREFERENCE PREDICTION COMPLETED 🚀")
print("=" * 70)

print(f"\nModel Accuracy : {accuracy * 100:.2f}%")
print(f"Model saved    : {model_file}")
print(f"Predictions    : {prediction_file}")
print(f"Feature import.: {importance_file}")

print("\nTop 10 Feature Importances:")

print(
    importance.head(10).to_string(
        index=False
    )
)

print("\nSample Predictions:")

print(
    df[
        [
            "customer_id",
            "favorite_cuisine",
            "predicted_cuisine"
        ]
    ].head(10).to_string(
        index=False
    )
)

print("\n" + "=" * 70)