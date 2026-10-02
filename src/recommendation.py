import os
import pandas as pd


# ============================================================
# FOODLENS - PERSONALIZED RECOMMENDATION SYSTEM
# ============================================================

print("=" * 70)
print("             FOODLENS RECOMMENDATION SYSTEM")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

print("\n[1/5] Loading customer and order data...")

customer_file = os.path.join(
    "data",
    "processed",
    "customer_predictions.csv"
)

orders_file = os.path.join(
    "data",
    "processed",
    "processed_orders.csv"
)

customers = pd.read_csv(customer_file)
orders = pd.read_csv(orders_file)

print(f"Customers loaded : {len(customers):,}")
print(f"Orders loaded    : {len(orders):,}")


# ------------------------------------------------------------
# 2. Create restaurant-cuisine information
# ------------------------------------------------------------

print("\n[2/5] Analyzing restaurant preferences...")

restaurant_data = (
    orders.groupby(
        ["restaurant_id", "cuisine"]
    )
    .agg(
        order_count=("order_id", "count"),
        average_rating=("rating", "mean"),
        average_order_value=("order_value", "mean")
    )
    .reset_index()
)

print(
    f"Restaurant-cuisine combinations: "
    f"{len(restaurant_data):,}"
)


# ------------------------------------------------------------
# 3. Create recommendation function
# ------------------------------------------------------------

def recommend_restaurants(customer_id, top_n=5):

    customer = customers[
        customers["customer_id"] == customer_id
    ]

    if customer.empty:
        print(f"\nCustomer {customer_id} not found.")
        return

    customer = customer.iloc[0]

    predicted_cuisine = customer["predicted_cuisine"]
    segment = customer.get(
        "customer_value_segment",
        "Regular Customer"
    )

    print("\n" + "-" * 70)
    print(f"Customer ID       : {customer_id}")
    print(f"Predicted Cuisine : {predicted_cuisine}")
    print(f"Customer Segment  : {segment}")
    print("-" * 70)

    # Restaurants serving predicted cuisine
    recommendations = restaurant_data[
        restaurant_data["cuisine"] == predicted_cuisine
    ].copy()

    if recommendations.empty:
        print("No recommendations available.")
        return

    # Recommendation score
    recommendations["score"] = (
        recommendations["average_rating"] * 0.50
        + recommendations["order_count"] /
        recommendations["order_count"].max() * 0.30
        + (
            1 -
            recommendations["average_order_value"] /
            recommendations["average_order_value"].max()
        ) * 0.20
    )

    recommendations = recommendations.sort_values(
        by="score",
        ascending=False
    )

    recommendations = recommendations.head(top_n)

    print("\n🍽️ Recommended Restaurants:\n")

    for rank, (_, row) in enumerate(
        recommendations.iterrows(),
        start=1
    ):
        print(
            f"{rank}. {row['restaurant_id']}"
        )
        print(
            f"   Cuisine        : {row['cuisine']}"
        )
        print(
            f"   Rating         : "
            f"{row['average_rating']:.2f}"
        )
        print(
            f"   Avg Order     : "
            f"₹{row['average_order_value']:.2f}"
        )
        print(
            f"   Recommendation Score: "
            f"{row['score']:.3f}"
        )
        print()


# ------------------------------------------------------------
# 4. Generate recommendations
# ------------------------------------------------------------

print("\n[3/5] Generating personalized recommendations...")

sample_customers = customers[
    "customer_id"
].head(5)

for customer_id in sample_customers:
    recommend_restaurants(
        customer_id,
        top_n=5
    )


# ------------------------------------------------------------
# 5. Save recommendation data
# ------------------------------------------------------------

print("\n[4/5] Creating recommendation dataset...")

recommendation_rows = []

for _, customer in customers.iterrows():

    customer_id = customer["customer_id"]
    predicted_cuisine = customer["predicted_cuisine"]

    matching_restaurants = restaurant_data[
        restaurant_data["cuisine"] == predicted_cuisine
    ].copy()

    if matching_restaurants.empty:
        continue

    matching_restaurants["score"] = (
        matching_restaurants["average_rating"] * 0.50
        + matching_restaurants["order_count"] /
        matching_restaurants["order_count"].max() * 0.30
        + (
            1 -
            matching_restaurants["average_order_value"] /
            matching_restaurants["average_order_value"].max()
        ) * 0.20
    )

    top_restaurants = matching_restaurants.sort_values(
        by="score",
        ascending=False
    ).head(5)

    for rank, (_, restaurant) in enumerate(
        top_restaurants.iterrows(),
        start=1
    ):

        recommendation_rows.append({
            "customer_id": customer_id,
            "customer_segment": customer.get(
                "customer_value_segment",
                "Regular Customer"
            ),
            "predicted_cuisine": predicted_cuisine,
            "rank": rank,
            "restaurant_id": restaurant["restaurant_id"],
            "restaurant_rating": round(
                restaurant["average_rating"],
                2
            ),
            "average_order_value": round(
                restaurant["average_order_value"],
                2
            ),
            "recommendation_score": round(
                restaurant["score"],
                3
            )
        })


recommendations_df = pd.DataFrame(
    recommendation_rows
)


output_file = os.path.join(
    "data",
    "processed",
    "recommendations.csv"
)

recommendations_df.to_csv(
    output_file,
    index=False
)


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("\n[5/5] Saving recommendations...")

print(
    f"✓ Recommendations generated: "
    f"{len(recommendations_df):,}"
)

print(
    f"✓ File saved: {output_file}"
)

print("\n" + "=" * 70)
print("       RECOMMENDATION SYSTEM COMPLETED 🚀")
print("=" * 70)