import os
import pandas as pd
import numpy as np


# ============================================================
# FOODLENS - BUSINESS INTELLIGENCE & ANALYTICS ENGINE
# ============================================================

print("=" * 70)
print("              FOODLENS BUSINESS ANALYTICS")
print("=" * 70)


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

INPUT_FILE = os.path.join(
    "data",
    "processed",
    "processed_orders.csv"
)

OUTPUT_DIR = os.path.join(
    "data",
    "processed"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. Load processed data
# ------------------------------------------------------------

print("\n[1/10] Loading processed data...")

df = pd.read_csv(INPUT_FILE)

df["order_date"] = pd.to_datetime(
    df["order_date"]
)

print(f"Records loaded: {len(df):,}")


# ------------------------------------------------------------
# 3. Basic business KPIs
# ------------------------------------------------------------

print("\n[2/10] Calculating business KPIs...")

delivered = df[
    df["order_status"] == "Delivered"
].copy()

total_orders = len(df)

delivered_orders = len(delivered)

cancelled_orders = len(
    df[df["order_status"] == "Cancelled"]
)

total_revenue = delivered[
    "final_amount"
].sum()

average_order_value = delivered[
    "final_amount"
].mean()

average_rating = delivered[
    "rating"
].mean()

average_delivery_time = delivered[
    "delivery_time_min"
].mean()

cancellation_rate = (
    cancelled_orders / total_orders
) * 100


print(f"Total Orders          : {total_orders:,}")
print(f"Delivered Orders      : {delivered_orders:,}")
print(f"Cancelled Orders      : {cancelled_orders:,}")
print(f"Total Revenue         : ₹{total_revenue:,.2f}")
print(f"Average Order Value   : ₹{average_order_value:,.2f}")
print(f"Average Rating        : {average_rating:.2f}")
print(
    f"Average Delivery Time : "
    f"{average_delivery_time:.2f} minutes"
)
print(
    f"Cancellation Rate     : "
    f"{cancellation_rate:.2f}%"
)


# ------------------------------------------------------------
# 4. Monthly performance
# ------------------------------------------------------------

print("\n[3/10] Analyzing monthly performance...")

monthly = (
    delivered
    .assign(
        month=delivered["order_date"].dt.to_period("M")
    )
    .groupby("month")
    .agg(
        orders=("order_id", "count"),
        revenue=("final_amount", "sum"),
        average_order_value=(
            "final_amount",
            "mean"
        ),
        average_rating=(
            "rating",
            "mean"
        )
    )
    .reset_index()
)

monthly["month"] = (
    monthly["month"]
    .astype(str)
)

monthly.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "monthly_performance.csv"
    ),
    index=False
)

print("\nTop Revenue Months:")

print(
    monthly
    .sort_values(
        "revenue",
        ascending=False
    )
    .head(5)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 5. City performance
# ------------------------------------------------------------

print("\n[4/10] Analyzing city performance...")

city_analysis = (
    delivered
    .groupby("city")
    .agg(
        orders=("order_id", "count"),
        revenue=("final_amount", "sum"),
        average_order_value=(
            "final_amount",
            "mean"
        ),
        average_delivery_time=(
            "delivery_time_min",
            "mean"
        ),
        average_rating=(
            "rating",
            "mean"
        )
    )
    .reset_index()
)

city_analysis["revenue_share"] = (
    city_analysis["revenue"]
    / city_analysis["revenue"].sum()
    * 100
)

city_analysis = city_analysis.sort_values(
    "revenue",
    ascending=False
)

city_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "city_analysis.csv"
    ),
    index=False
)

print("\nCity Performance:")

print(
    city_analysis.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 6. Cuisine intelligence
# ------------------------------------------------------------

print("\n[5/10] Analyzing cuisine preferences...")

cuisine_analysis = (
    delivered
    .groupby("cuisine")
    .agg(
        orders=("order_id", "count"),
        revenue=("final_amount", "sum"),
        average_order_value=(
            "final_amount",
            "mean"
        ),
        average_rating=(
            "rating",
            "mean"
        ),
        average_delivery_time=(
            "delivery_time_min",
            "mean"
        )
    )
    .reset_index()
)

cuisine_analysis["order_share"] = (
    cuisine_analysis["orders"]
    / cuisine_analysis["orders"].sum()
    * 100
)

cuisine_analysis = cuisine_analysis.sort_values(
    "orders",
    ascending=False
)

cuisine_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cuisine_analysis.csv"
    ),
    index=False
)

print("\nCuisine Performance:")

print(
    cuisine_analysis.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 7. Peak ordering analysis
# ------------------------------------------------------------

print("\n[6/10] Finding peak ordering periods...")

hourly_analysis = (
    delivered
    .groupby("hour")
    .agg(
        orders=("order_id", "count"),
        revenue=("final_amount", "sum")
    )
    .reset_index()
)

hourly_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "hourly_analysis.csv"
    ),
    index=False
)

peak_hour = (
    hourly_analysis
    .sort_values(
        "orders",
        ascending=False
    )
    .iloc[0]
)

print(
    f"Peak Hour: {int(peak_hour['hour']):02d}:00"
)

print(
    f"Orders during peak hour: "
    f"{int(peak_hour['orders']):,}"
)


# ------------------------------------------------------------
# 8. Meal period analysis
# ------------------------------------------------------------

meal_analysis = (
    delivered
    .groupby("meal_period")
    .agg(
        orders=("order_id", "count"),
        revenue=("final_amount", "sum"),
        average_order_value=(
            "final_amount",
            "mean"
        ),
        average_rating=(
            "rating",
            "mean"
        )
    )
    .reset_index()
)

meal_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "meal_period_analysis.csv"
    ),
    index=False
)

print("\nMeal Period Performance:")

print(
    meal_analysis
    .sort_values(
        "orders",
        ascending=False
    )
    .to_string(index=False)
)


# ------------------------------------------------------------
# 9. Weekend vs weekday
# ------------------------------------------------------------

print("\n[7/10] Comparing weekend and weekday behavior...")

weekend_analysis = (
    delivered
    .groupby("is_weekend")
    .agg(
        orders=("order_id", "count"),
        revenue=("final_amount", "sum"),
        average_order_value=(
            "final_amount",
            "mean"
        ),
        average_rating=(
            "rating",
            "mean"
        )
    )
    .reset_index()
)

weekend_analysis[
    "day_type"
] = np.where(
    weekend_analysis["is_weekend"],
    "Weekend",
    "Weekday"
)

weekend_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "weekend_analysis.csv"
    ),
    index=False
)

print(
    weekend_analysis[
        [
            "day_type",
            "orders",
            "revenue",
            "average_order_value",
            "average_rating"
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# 10. Delivery & discount intelligence
# ------------------------------------------------------------

print("\n[8/10] Analyzing delivery performance...")

delivery_analysis = (
    delivered
    .groupby("delivery_category")
    .agg(
        orders=("order_id", "count"),
        average_delivery_time=(
            "delivery_time_min",
            "mean"
        ),
        average_rating=(
            "rating",
            "mean"
        ),
        average_order_value=(
            "final_amount",
            "mean"
        )
    )
    .reset_index()
)

delivery_order = [
    "Fast",
    "Normal",
    "Slow",
    "Very Slow"
]

delivery_analysis[
    "sort_order"
] = delivery_analysis[
    "delivery_category"
].map(
    {
        category: i
        for i, category
        in enumerate(delivery_order)
    }
)

delivery_analysis = (
    delivery_analysis
    .sort_values("sort_order")
    .drop(columns="sort_order")
)

delivery_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "delivery_analysis.csv"
    ),
    index=False
)

print(
    delivery_analysis.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Discount effectiveness
# ------------------------------------------------------------

print("\n[9/10] Analyzing discount effectiveness...")

discount_analysis = (
    delivered
    .groupby("discount_percent")
    .agg(
        orders=("order_id", "count"),
        revenue=("final_amount", "sum"),
        average_order_value=(
            "final_amount",
            "mean"
        ),
        average_rating=(
            "rating",
            "mean"
        )
    )
    .reset_index()
)

discount_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "discount_analysis.csv"
    ),
    index=False
)

print(
    discount_analysis.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Weather impact
# ------------------------------------------------------------

weather_analysis = (
    delivered
    .groupby("weather")
    .agg(
        orders=("order_id", "count"),
        revenue=("final_amount", "sum"),
        average_delivery_time=(
            "delivery_time_min",
            "mean"
        ),
        average_rating=(
            "rating",
            "mean"
        )
    )
    .reset_index()
)

weather_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "weather_analysis.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# Restaurant performance
# ------------------------------------------------------------

print("\n[10/10] Analyzing restaurant performance...")

restaurant_analysis = (
    delivered
    .groupby("restaurant_id")
    .agg(
        orders=("order_id", "count"),
        revenue=("final_amount", "sum"),
        average_order_value=(
            "final_amount",
            "mean"
        ),
        average_rating=(
            "rating",
            "mean"
        ),
        average_delivery_time=(
            "delivery_time_min",
            "mean"
        )
    )
    .reset_index()
)

restaurant_analysis = (
    restaurant_analysis
    .sort_values(
        "revenue",
        ascending=False
    )
)

restaurant_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "restaurant_analysis.csv"
    ),
    index=False
)


# ============================================================
# FINAL INSIGHTS
# ============================================================

print("\n" + "=" * 70)
print("                 KEY FOODLENS INSIGHTS")
print("=" * 70)


top_city = city_analysis.iloc[0]

top_cuisine = cuisine_analysis.iloc[0]

top_restaurant = restaurant_analysis.iloc[0]

fastest_category = (
    delivery_analysis
    .sort_values(
        "average_delivery_time"
    )
    .iloc[0]
)


print(
    f"\n🏙️ Top Revenue City:"
    f" {top_city['city']}"
)

print(
    f"   Revenue: ₹{top_city['revenue']:,.2f}"
)


print(
    f"\n🍛 Most Ordered Cuisine:"
    f" {top_cuisine['cuisine']}"
)

print(
    f"   Orders: {int(top_cuisine['orders']):,}"
)


print(
    f"\n🏪 Top Restaurant by Revenue:"
    f" {top_restaurant['restaurant_id']}"
)

print(
    f"   Revenue: ₹{top_restaurant['revenue']:,.2f}"
)


print(
    f"\n🚚 Fastest Delivery Category:"
    f" {fastest_category['delivery_category']}"
)

print(
    f"   Average Time:"
    f" {fastest_category['average_delivery_time']:.2f} min"
)


print(
    f"\n⏰ Peak Ordering Hour:"
    f" {int(peak_hour['hour']):02d}:00"
)


print("\n📁 Analysis files generated:")

analysis_files = [
    "monthly_performance.csv",
    "city_analysis.csv",
    "cuisine_analysis.csv",
    "hourly_analysis.csv",
    "meal_period_analysis.csv",
    "weekend_analysis.csv",
    "delivery_analysis.csv",
    "discount_analysis.csv",
    "weather_analysis.csv",
    "restaurant_analysis.csv"
]

for file in analysis_files:
    print(f"   ✓ {file}")


print("\n" + "=" * 70)
print("          FOODLENS ANALYTICS COMPLETED 🚀")
print("=" * 70)