import os
import pandas as pd
import numpy as np


# ============================================================
# FOODLENS - DATA PREPROCESSING PIPELINE
# ============================================================

print("=" * 65)
print("              FOODLENS DATA PREPROCESSING")
print("=" * 65)


# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

INPUT_FILE = os.path.join(
    "data",
    "raw",
    "food_delivery_orders.csv"
)

PROCESSED_DIR = os.path.join(
    "data",
    "processed"
)

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------------

print("\n[1/7] Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ------------------------------------------------------------
# 3. Data quality checks
# ------------------------------------------------------------

print("\n[2/7] Running data quality checks...")

missing_values = df.isnull().sum().sum()
duplicate_rows = df.duplicated().sum()

print(f"Missing values : {missing_values:,}")
print(f"Duplicate rows : {duplicate_rows:,}")

if missing_values == 0:
    print("✓ No missing values found")

if duplicate_rows == 0:
    print("✓ No duplicate rows found")


# ------------------------------------------------------------
# 4. Data type conversion
# ------------------------------------------------------------

print("\n[3/7] Converting data types...")

df["order_date"] = pd.to_datetime(
    df["order_date"]
)

df["order_time"] = pd.to_datetime(
    df["order_time"],
    format="%H:%M:%S"
).dt.time


# ------------------------------------------------------------
# 5. Feature engineering
# ------------------------------------------------------------

print("\n[4/7] Creating analytical features...")


# Date features

df["year"] = df["order_date"].dt.year

df["month"] = df["order_date"].dt.month

df["day_of_week"] = df["order_date"].dt.day_name()

df["day_of_month"] = df["order_date"].dt.day

df["week_of_year"] = (
    df["order_date"].dt.isocalendar().week
)


# Time feature

df["hour"] = pd.to_datetime(
    df["order_time"].astype(str)
).dt.hour


# Meal period

def get_meal_period(hour):

    if 6 <= hour <= 10:
        return "Breakfast"

    elif 11 <= hour <= 15:
        return "Lunch"

    elif 16 <= hour <= 18:
        return "Evening Snack"

    elif 19 <= hour <= 22:
        return "Dinner"

    else:
        return "Late Night"


df["meal_period"] = df["hour"].apply(
    get_meal_period
)


# Weekend indicator

df["is_weekend"] = df["day_of_week"].isin(
    ["Saturday", "Sunday"]
)


# Final amount after discount

df["final_amount"] = (
    df["order_value"]
    - df["discount_amount"]
    + df["delivery_fee"]
)


# Discount ratio

df["discount_ratio"] = np.where(
    df["order_value"] > 0,
    df["discount_amount"] / df["order_value"],
    0
)


# Delivery category

def delivery_category(minutes):

    if minutes <= 25:
        return "Fast"

    elif minutes <= 40:
        return "Normal"

    elif minutes <= 55:
        return "Slow"

    else:
        return "Very Slow"


df["delivery_category"] = df[
    "delivery_time_min"
].apply(delivery_category)


# Order value category

def order_value_category(value):

    if value < 250:
        return "Low"

    elif value < 500:
        return "Medium"

    elif value < 1000:
        return "High"

    else:
        return "Premium"


df["order_value_category"] = df[
    "order_value"
].apply(order_value_category)


print("✓ Date features created")
print("✓ Time features created")
print("✓ Meal period created")
print("✓ Financial features created")
print("✓ Delivery categories created")


# ------------------------------------------------------------
# 6. Create customer-level analytical dataset
# ------------------------------------------------------------

print("\n[5/7] Building customer profiles...")


# Only delivered orders should influence
# customer behavior analysis.

delivered_df = df[
    df["order_status"] == "Delivered"
].copy()


customer_features = (
    delivered_df
    .groupby("customer_id")
    .agg(
        total_orders=("order_id", "count"),

        average_order_value=(
            "order_value",
            "mean"
        ),

        total_spending=(
            "final_amount",
            "sum"
        ),

        average_rating=(
            "rating",
            "mean"
        ),

        average_delivery_time=(
            "delivery_time_min",
            "mean"
        ),

        average_discount=(
            "discount_percent",
            "mean"
        ),

        average_distance=(
            "distance_km",
            "mean"
        ),

        average_quantity=(
            "quantity",
            "mean"
        ),

        average_delivery_fee=(
            "delivery_fee",
            "mean"
        )
    )
    .reset_index()
)


# ------------------------------------------------------------
# Favorite cuisine
# ------------------------------------------------------------

favorite_cuisine = (
    delivered_df
    .groupby(
        ["customer_id", "cuisine"]
    )
    .size()
    .reset_index(name="order_count")
)


favorite_cuisine = (
    favorite_cuisine
    .sort_values(
        ["customer_id", "order_count"],
        ascending=[True, False]
    )
    .drop_duplicates(
        "customer_id"
    )
    [["customer_id", "cuisine"]]
    .rename(
        columns={
            "cuisine": "favorite_cuisine"
        }
    )
)


customer_features = customer_features.merge(
    favorite_cuisine,
    on="customer_id",
    how="left"
)


# ------------------------------------------------------------
# Customer frequency category
# ------------------------------------------------------------

def frequency_category(orders):

    if orders <= 5:
        return "Occasional"

    elif orders <= 15:
        return "Regular"

    elif orders <= 30:
        return "Frequent"

    else:
        return "Very Frequent"


customer_features[
    "customer_frequency"
] = customer_features[
    "total_orders"
].apply(frequency_category)


# ------------------------------------------------------------
# Customer spending category
# ------------------------------------------------------------

def spending_category(spending):

    if spending < 5000:
        return "Low Value"

    elif spending < 15000:
        return "Medium Value"

    elif spending < 30000:
        return "High Value"

    else:
        return "Premium"


customer_features[
    "customer_value_segment"
] = customer_features[
    "total_spending"
].apply(spending_category)


print(
    f"✓ Customer profiles created: "
    f"{len(customer_features):,}"
)


# ------------------------------------------------------------
# 7. Save processed datasets
# ------------------------------------------------------------

print("\n[6/7] Saving processed datasets...")


orders_output = os.path.join(
    PROCESSED_DIR,
    "processed_orders.csv"
)

customers_output = os.path.join(
    PROCESSED_DIR,
    "customer_features.csv"
)


df.to_csv(
    orders_output,
    index=False
)


customer_features.to_csv(
    customers_output,
    index=False
)


print(
    f"✓ Orders saved    : {orders_output}"
)

print(
    f"✓ Customers saved : {customers_output}"
)


# ------------------------------------------------------------
# 8. Final report
# ------------------------------------------------------------

print("\n[7/7] PREPROCESSING SUMMARY")

print("-" * 65)

print(
    f"Original records       : {len(df):,}"
)

print(
    f"Delivered records      : {len(delivered_df):,}"
)

print(
    f"Customer profiles      : "
    f"{len(customer_features):,}"
)

print(
    f"Final order features   : "
    f"{len(df.columns)}"
)

print(
    f"Customer features      : "
    f"{len(customer_features.columns)}"
)


print("\nCustomer Frequency:")
print(
    customer_features[
        "customer_frequency"
    ].value_counts()
)


print("\nCustomer Value Segments:")
print(
    customer_features[
        "customer_value_segment"
    ].value_counts()
)


print("\nTop 10 Favorite Cuisines:")
print(
    customer_features[
        "favorite_cuisine"
    ].value_counts()
    .head(10)
)


print("\nSample Customer Profiles:")

print(
    customer_features.head(10).to_string(
        index=False
    )
)


print("\n" + "=" * 65)
print("        PREPROCESSING COMPLETED SUCCESSFULLY 🚀")
print("=" * 65)
