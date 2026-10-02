import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# ============================================================
# FOODLENS - CUSTOMER SEGMENTATION
# ============================================================

print("=" * 70)
print("             FOODLENS CUSTOMER SEGMENTATION")
print("=" * 70)


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

INPUT_FILE = os.path.join(
    "data",
    "processed",
    "customer_features.csv"
)

OUTPUT_DIR = os.path.join(
    "data",
    "processed"
)

MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. Load customer data
# ------------------------------------------------------------

print("\n[1/7] Loading customer profiles...")

df = pd.read_csv(INPUT_FILE)

print(f"Customers loaded: {len(df):,}")


# ------------------------------------------------------------
# 3. Select behavioral features
# ------------------------------------------------------------

print("\n[2/7] Selecting customer behavior features...")

features = [
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

X = df[features].copy()

print("\nFeatures used for clustering:")

for feature in features:
    print(f"   ✓ {feature}")


# ------------------------------------------------------------
# 4. Standardize features
# ------------------------------------------------------------

print("\n[3/7] Standardizing features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("✓ Feature scaling completed")


# ------------------------------------------------------------
# 5. Find optimal number of clusters
# ------------------------------------------------------------

print("\n[4/7] Finding optimal number of clusters...")

inertia_values = []
silhouette_values = []

cluster_range = range(2, 9)

for k in cluster_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        X_scaled
    )

    inertia_values.append(
        model.inertia_
    )

    silhouette_values.append(
        silhouette_score(
            X_scaled,
            labels
        )
    )

    print(
        f"K={k} | "
        f"Inertia={model.inertia_:.2f} | "
        f"Silhouette={silhouette_values[-1]:.4f}"
    )


# ------------------------------------------------------------
# Save cluster evaluation
# ------------------------------------------------------------

evaluation_df = pd.DataFrame({
    "k": list(cluster_range),
    "inertia": inertia_values,
    "silhouette_score": silhouette_values
})

evaluation_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cluster_evaluation.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# Select best K using silhouette score
# ------------------------------------------------------------

best_index = np.argmax(
    silhouette_values
)

best_k = list(cluster_range)[
    best_index
]

best_silhouette = silhouette_values[
    best_index
]

print(
    f"\n✓ Best K based on silhouette score: "
    f"{best_k}"
)

print(
    f"✓ Best silhouette score: "
    f"{best_silhouette:.4f}"
)


# ------------------------------------------------------------
# 6. Train final K-Means model
# ------------------------------------------------------------

print("\n[5/7] Training final K-Means model...")

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(
    X_scaled
)

print("✓ Customer clusters created")


# ------------------------------------------------------------
# 7. Profile clusters
# ------------------------------------------------------------

print("\n[6/7] Profiling customer segments...")

cluster_profile = (
    df
    .groupby("cluster")[features]
    .mean()
    .round(2)
)

cluster_counts = (
    df["cluster"]
    .value_counts()
    .sort_index()
    .rename("customer_count")
)

cluster_profile = cluster_profile.join(
    cluster_counts
)

cluster_profile["percentage"] = (
    cluster_profile["customer_count"]
    / len(df)
    * 100
)

cluster_profile.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cluster_profiles.csv"
    )
)


# ------------------------------------------------------------
# Print profiles
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("CUSTOMER CLUSTER PROFILES")
print("-" * 70)

print(
    cluster_profile.to_string()
)


# ------------------------------------------------------------
# Automatically generate segment descriptions
# ------------------------------------------------------------

overall_means = df[features].mean()


def describe_cluster(
    cluster_row,
    overall
):

    descriptions = []

    if (
        cluster_row["total_orders"]
        > overall["total_orders"] * 1.30
    ):
        descriptions.append(
            "Highly Frequent"
        )

    elif (
        cluster_row["total_orders"]
        < overall["total_orders"] * 0.70
    ):
        descriptions.append(
            "Occasional"
        )

    if (
        cluster_row["total_spending"]
        > overall["total_spending"] * 1.30
    ):
        descriptions.append(
            "High Spending"
        )

    elif (
        cluster_row["total_spending"]
        < overall["total_spending"] * 0.70
    ):
        descriptions.append(
            "Low Spending"
        )

    if (
        cluster_row["average_discount"]
        > overall["average_discount"] * 1.25
    ):
        descriptions.append(
            "Discount Sensitive"
        )

    if (
        cluster_row["average_order_value"]
        > overall["average_order_value"] * 1.20
    ):
        descriptions.append(
            "High Basket Value"
        )

    if (
        cluster_row["average_delivery_time"]
        > overall["average_delivery_time"] * 1.20
    ):
        descriptions.append(
            "Long Delivery Exposure"
        )

    if not descriptions:
        descriptions.append(
            "Balanced Customers"
        )

    return " / ".join(descriptions)


cluster_profile[
    "segment_description"
] = cluster_profile.apply(
    lambda row: describe_cluster(
        row,
        overall_means
    ),
    axis=1
)


# ------------------------------------------------------------
# Print segment descriptions
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("INTERPRETED CUSTOMER SEGMENTS")
print("-" * 70)

for cluster_id, row in cluster_profile.iterrows():

    print(
        f"\nCluster {cluster_id}"
    )

    print(
        f"Customers: "
        f"{int(row['customer_count']):,} "
        f"({row['percentage']:.1f}%)"
    )

    print(
        f"Profile: "
        f"{row['segment_description']}"
    )


# ------------------------------------------------------------
# Save final customer segmentation
# ------------------------------------------------------------

segmented_file = os.path.join(
    OUTPUT_DIR,
    "customer_segments.csv"
)

df.to_csv(
    segmented_file,
    index=False
)


# ------------------------------------------------------------
# Save scaler and model parameters
# ------------------------------------------------------------

np.save(
    os.path.join(
        MODEL_DIR,
        "customer_scaler_mean.npy"
    ),
    scaler.mean_
)

np.save(
    os.path.join(
        MODEL_DIR,
        "customer_scaler_scale.npy"
    ),
    scaler.scale_
)

np.save(
    os.path.join(
        MODEL_DIR,
        "kmeans_cluster_centers.npy"
    ),
    kmeans.cluster_centers_
)


# ------------------------------------------------------------
# Visualization 1: Elbow Curve
# ------------------------------------------------------------

print("\n[7/7] Creating cluster visualizations...")

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    list(cluster_range),
    inertia_values,
    marker="o"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Within-Cluster Sum of Squares"
)

plt.title(
    "FoodLens - K-Means Elbow Analysis"
)

plt.xticks(
    list(cluster_range)
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "elbow_curve.png"
    ),
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# Visualization 2: Silhouette Score
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    list(cluster_range),
    silhouette_values,
    marker="o"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Silhouette Score"
)

plt.title(
    "FoodLens - Silhouette Score Analysis"
)

plt.xticks(
    list(cluster_range)
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "silhouette_analysis.png"
    ),
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# Final output
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("        CUSTOMER SEGMENTATION COMPLETED 🚀")
print("=" * 70)

print(
    f"\nOptimal clusters : {best_k}"
)

print(
    f"Silhouette score : "
    f"{best_silhouette:.4f}"
)

print(
    f"Segmented users  : "
    f"{len(df):,}"
)

print("\nFiles generated:")

files = [
    "cluster_evaluation.csv",
    "cluster_profiles.csv",
    "customer_segments.csv",
    "elbow_curve.png",
    "silhouette_analysis.png"
]

for file in files:
    print(f"   ✓ data/processed/{file}")

print("\n" + "=" * 70)