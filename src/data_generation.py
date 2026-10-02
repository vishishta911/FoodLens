import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ============================================================
# FOODLENS - REALISTIC FOOD DELIVERY DATA GENERATOR
# ============================================================

print("=" * 60)
print("             FOODLENS DATA GENERATOR")
print("=" * 60)


# -----------------------------
# Configuration
# -----------------------------

NUM_ORDERS = 200_000
NUM_CUSTOMERS = 10_000
NUM_RESTAURANTS = 1_000

OUTPUT_DIR = os.path.join("data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "food_delivery_orders.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)


# -----------------------------
# Master data
# -----------------------------

cities = [
    "Hyderabad",
    "Warangal",
    "Secunderabad",
    "Karimnagar",
    "Nizamabad",
    "Vijayawada",
    "Visakhapatnam",
    "Bengaluru",
    "Chennai",
    "Pune",
    "Mumbai"
]

cuisines = [
    "Biryani",
    "Indian",
    "South Indian",
    "North Indian",
    "Chinese",
    "Fast Food",
    "Italian",
    "Mexican",
    "Healthy",
    "Desserts"
]

payment_methods = [
    "UPI",
    "Credit Card",
    "Debit Card",
    "Cash on Delivery",
    "Wallet"
]

weather_conditions = [
    "Clear",
    "Cloudy",
    "Rainy",
    "Hot",
    "Stormy"
]

genders = [
    "Male",
    "Female",
    "Other"
]

order_statuses = [
    "Delivered",
    "Delivered",
    "Delivered",
    "Delivered",
    "Cancelled"
]

meal_periods = [
    "Breakfast",
    "Lunch",
    "Evening Snack",
    "Dinner",
    "Late Night"
]


# -----------------------------
# Restaurant information
# -----------------------------

restaurant_ids = [
    f"R{str(i).zfill(5)}"
    for i in range(1, NUM_RESTAURANTS + 1)
]

restaurant_city = np.random.choice(
    cities,
    NUM_RESTAURANTS
)

restaurant_cuisine = np.random.choice(
    cuisines,
    NUM_RESTAURANTS
)


# -----------------------------
# Customer information
# -----------------------------

customer_ids = [
    f"C{str(i).zfill(5)}"
    for i in range(1, NUM_CUSTOMERS + 1)
]

customer_age = np.random.randint(
    18,
    61,
    NUM_CUSTOMERS
)

customer_gender = np.random.choice(
    genders,
    NUM_CUSTOMERS,
    p=[0.48, 0.49, 0.03]
)

customer_city = np.random.choice(
    cities,
    NUM_CUSTOMERS
)


# ---------------------------------------------------------
# Customer behavioral profiles
# ---------------------------------------------------------
#
# We intentionally create different customer behaviors.
# K-Means will later discover these patterns without
# being given the profile labels.
# ---------------------------------------------------------

customer_profiles = [
    "Budget Hunter",
    "Premium Foodie",
    "Frequent Foodie",
    "Occasional Customer",
    "Healthy Customer",
    "Late Night Customer"
]

profile_probabilities = [
    0.20,
    0.15,
    0.20,
    0.15,
    0.15,
    0.15
]


customer_info = {}


for i, customer_id in enumerate(customer_ids):

    profile = np.random.choice(
        customer_profiles,
        p=profile_probabilities
    )

    age = customer_age[i]
    gender = customer_gender[i]
    city = customer_city[i]

    # ---------------------------------------------
    # Profile-specific behavior
    # ---------------------------------------------

    if profile == "Budget Hunter":

        preferred_cuisine = np.random.choice(
            ["Fast Food", "Chinese", "South Indian"],
            p=[0.40, 0.30, 0.30]
        )

        order_frequency = np.random.uniform(
            1.2, 2.2
        )

        discount_sensitivity = np.random.uniform(
            0.75, 1.0
        )

        spending_factor = np.random.uniform(
            0.65, 0.90
        )

        quantity_factor = np.random.uniform(
            1.0, 1.5
        )

    elif profile == "Premium Foodie":

        preferred_cuisine = np.random.choice(
            ["Italian", "Mexican", "Biryani", "Healthy"],
            p=[0.30, 0.25, 0.25, 0.20]
        )

        order_frequency = np.random.uniform(
            1.3, 2.3
        )

        discount_sensitivity = np.random.uniform(
            0.10, 0.40
        )

        spending_factor = np.random.uniform(
            1.30, 1.80
        )

        quantity_factor = np.random.uniform(
            1.2, 1.8
        )

    elif profile == "Frequent Foodie":

        preferred_cuisine = np.random.choice(
            cuisines
        )

        order_frequency = np.random.uniform(
            2.0, 3.5
        )

        discount_sensitivity = np.random.uniform(
            0.30, 0.70
        )

        spending_factor = np.random.uniform(
            0.95, 1.30
        )

        quantity_factor = np.random.uniform(
            1.4, 2.2
        )

    elif profile == "Occasional Customer":

        preferred_cuisine = np.random.choice(
            cuisines
        )

        order_frequency = np.random.uniform(
            0.3, 0.9
        )

        discount_sensitivity = np.random.uniform(
            0.30, 0.80
        )

        spending_factor = np.random.uniform(
            0.70, 1.05
        )

        quantity_factor = np.random.uniform(
            0.8, 1.3
        )

    elif profile == "Healthy Customer":

        preferred_cuisine = np.random.choice(
            ["Healthy", "South Indian", "Indian"],
            p=[0.60, 0.25, 0.15]
        )

        order_frequency = np.random.uniform(
            1.0, 2.2
        )

        discount_sensitivity = np.random.uniform(
            0.20, 0.60
        )

        spending_factor = np.random.uniform(
            0.90, 1.25
        )

        quantity_factor = np.random.uniform(
            1.0, 1.6
        )

    else:
        # Late Night Customer

        preferred_cuisine = np.random.choice(
            ["Fast Food", "Chinese", "Desserts", "Biryani"],
            p=[0.30, 0.25, 0.20, 0.25]
        )

        order_frequency = np.random.uniform(
            1.0, 2.5
        )

        discount_sensitivity = np.random.uniform(
            0.30, 0.75
        )

        spending_factor = np.random.uniform(
            0.85, 1.25
        )

        quantity_factor = np.random.uniform(
            1.1, 1.8
        )


    customer_info[customer_id] = {

        "age": age,

        "gender": gender,

        "city": city,

        "profile": profile,

        "preferred_cuisine": preferred_cuisine,

        "discount_sensitivity":
            discount_sensitivity,

        "order_frequency":
            order_frequency,

        "spending_factor":
            spending_factor,

        "quantity_factor":
            quantity_factor
    }


# Customer selection weights
customer_weights = np.array([
    customer_info[c]["order_frequency"]
    for c in customer_ids
])

customer_weights = customer_weights / customer_weights.sum()

# -----------------------------
# Food items
# -----------------------------

food_items = {
    "Biryani": [
        "Chicken Biryani",
        "Mutton Biryani",
        "Veg Biryani",
        "Paneer Biryani"
    ],
    "Indian": [
        "Butter Chicken",
        "Paneer Butter Masala",
        "Dal Tadka",
        "Veg Thali"
    ],
    "South Indian": [
        "Masala Dosa",
        "Idli",
        "Vada",
        "South Indian Meals"
    ],
    "North Indian": [
        "Chole Bhature",
        "Rajma Rice",
        "Paratha",
        "North Indian Thali"
    ],
    "Chinese": [
        "Fried Rice",
        "Hakka Noodles",
        "Manchurian",
        "Schezwan Rice"
    ],
    "Fast Food": [
        "Burger",
        "Pizza",
        "French Fries",
        "Sandwich"
    ],
    "Italian": [
        "Margherita Pizza",
        "Pasta",
        "Lasagna",
        "Garlic Bread"
    ],
    "Mexican": [
        "Tacos",
        "Burrito",
        "Nachos",
        "Quesadilla"
    ],
    "Healthy": [
        "Salad Bowl",
        "Protein Bowl",
        "Grilled Chicken",
        "Fruit Bowl"
    ],
    "Desserts": [
        "Brownie",
        "Ice Cream",
        "Gulab Jamun",
        "Cheesecake"
    ]
}


# -----------------------------
# Generate orders
# -----------------------------

print("\nGenerating orders...")

orders = []

start_date = datetime(2025, 1, 1)
end_date = datetime(2026, 9, 30)

date_range_days = (end_date - start_date).days


for i in range(NUM_ORDERS):

    # ---------------------------------
    # Customer
    # ---------------------------------

    customer_id = np.random.choice(
        customer_ids,
        p=customer_weights
    )
    customer = customer_info[customer_id]

    # ---------------------------------
    # Date
    # ---------------------------------

    order_date = start_date + timedelta(
        days=random.randint(0, date_range_days)
    )

    # ---------------------------------
    # Meal period
    # ---------------------------------

    meal_period = random.choices(
        meal_periods,
        weights=[0.10, 0.27, 0.18, 0.35, 0.10]
    )[0]

    meal_hours = {
        "Breakfast": (7, 10),
        "Lunch": (11, 15),
        "Evening Snack": (16, 18),
        "Dinner": (19, 22),
        "Late Night": (23, 23)
    }

    hour_start, hour_end = meal_hours[meal_period]

    hour = random.randint(hour_start, hour_end)

    minute = random.randint(0, 59)

    order_time = f"{hour:02d}:{minute:02d}:00"

    # ---------------------------------
    # Weekend
    # ---------------------------------

    is_weekend = order_date.weekday() >= 5

    # ---------------------------------
    # Cuisine
    #
    # 65% chance of customer's
    # preferred cuisine.
    # ---------------------------------

    if random.random() < 0.65:
        cuisine = customer["preferred_cuisine"]
    else:
        cuisine = random.choice(cuisines)

    # ---------------------------------
    # Food item
    # ---------------------------------

    item_name = random.choice(
        food_items[cuisine]
    )

    # ---------------------------------
    # Quantity
    # ---------------------------------

    quantity = int(
        np.clip(
            np.random.normal(
                1.5 * customer["quantity_factor"],
                0.7
            ),
            1,
            5
        )
    )

    # ---------------------------------
    # Base order value
    # ---------------------------------

    base_prices = {
        "Biryani": 350,
        "Indian": 300,
        "South Indian": 180,
        "North Indian": 280,
        "Chinese": 260,
        "Fast Food": 240,
        "Italian": 420,
        "Mexican": 380,
        "Healthy": 320,
        "Desserts": 180
    }

    price = np.random.normal(
        base_prices[cuisine]
        * customer["spending_factor"],
        base_prices[cuisine] * 0.15
    )

    price = max(price, 80)

    order_value = price * quantity

    # ---------------------------------
    # Discount
    #
    # Discount-sensitive customers
    # receive/use larger discounts.
    # ---------------------------------

    discount_probability = (
        0.20 +
        customer["discount_sensitivity"] * 0.45
    )

    if random.random() < discount_probability:

        discount_percent = random.choice(
            [5, 10, 15, 20, 25, 30]
        )

    else:

        discount_percent = 0

    discount_amount = (
        order_value * discount_percent / 100
    )

    # ---------------------------------
    # Delivery distance
    # ---------------------------------

    distance_km = round(
        np.random.gamma(
            shape=2.0,
            scale=1.8
        ),
        2
    )

    distance_km = min(
        max(distance_km, 0.5),
        15
    )

    # ---------------------------------
    # Weather
    # ---------------------------------

    weather = random.choices(
        weather_conditions,
        weights=[0.40, 0.22, 0.18, 0.15, 0.05]
    )[0]

    # ---------------------------------
    # Delivery time
    #
    # Distance + weather affect time.
    # ---------------------------------

    delivery_time = (
        20
        + distance_km * 3
        + random.gauss(0, 5)
    )

    if weather == "Rainy":
        delivery_time += 10

    elif weather == "Stormy":
        delivery_time += 18

    elif weather == "Hot":
        delivery_time += 3

    if is_weekend:
        delivery_time += 5

    delivery_time = int(
        max(delivery_time, 10)
    )

    # ---------------------------------
    # Delivery partner rating
    # ---------------------------------

    partner_rating = round(
        np.clip(
            np.random.normal(4.2, 0.45),
            2.5,
            5.0
        ),
        1
    )

    # ---------------------------------
    # Customer rating
    #
    # Longer delivery → slightly lower
    # rating.
    # ---------------------------------

    rating = (
        4.8
        - (delivery_time - 25) * 0.025
        + random.gauss(0, 0.35)
    )

    rating = round(
        np.clip(rating, 1, 5),
        1
    )

    # ---------------------------------
    # Order status
    # ---------------------------------

    cancellation_probability = 0.03

    if delivery_time > 55:
        cancellation_probability += 0.04

    if weather == "Stormy":
        cancellation_probability += 0.05

    if random.random() < cancellation_probability:
        order_status = "Cancelled"
    else:
        order_status = "Delivered"

    # ---------------------------------
    # Restaurant
    # ---------------------------------

    restaurant_id = random.choice(
        restaurant_ids
    )

    restaurant_index = int(
        restaurant_id[1:]
    ) - 1

    city = restaurant_city[
        restaurant_index
    ]

    # ---------------------------------
    # Previous orders
    # ---------------------------------

    previous_orders = int(
        np.random.gamma(
            shape=max(customer["order_frequency"], 0.5),
            scale=4
        )
    )

    previous_orders = min(
        previous_orders,
        60
    )

    # ---------------------------------
    # Payment method
    # ---------------------------------

    payment_method = random.choices(
        payment_methods,
        weights=[
            0.50,
            0.18,
            0.15,
            0.10,
            0.07
        ]
    )[0]

    # ---------------------------------
    # Store record
    # ---------------------------------

    orders.append([
        i + 1,
        customer_id,
        restaurant_id,
        order_date.strftime("%Y-%m-%d"),
        order_time,
        city,
        customer["age"],
        customer["gender"],
        cuisine,
        item_name,
        quantity,
        round(order_value, 2),
        discount_percent,
        round(discount_amount, 2),
        round(35 + distance_km * 4, 2),
        delivery_time,
        payment_method,
        rating,
        order_status,
        distance_km,
        weather,
        is_weekend,
        previous_orders,
        partner_rating
    ])

    # Progress
    if (i + 1) % 25_000 == 0:
        print(
            f"Generated {i + 1:,} / {NUM_ORDERS:,} orders"
        )


# -----------------------------
# Create DataFrame
# -----------------------------

columns = [
    "order_id",
    "customer_id",
    "restaurant_id",
    "order_date",
    "order_time",
    "city",
    "customer_age",
    "customer_gender",
    "cuisine",
    "item_name",
    "quantity",
    "order_value",
    "discount_percent",
    "discount_amount",
    "delivery_fee",
    "delivery_time_min",
    "payment_method",
    "rating",
    "order_status",
    "distance_km",
    "weather",
    "is_weekend",
    "previous_orders",
    "delivery_partner_rating"
]

df = pd.DataFrame(
    orders,
    columns=columns
)


# -----------------------------
# Save dataset
# -----------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# -----------------------------
# Summary
# -----------------------------

print("\n" + "=" * 60)
print("              DATASET CREATED")
print("=" * 60)

print(f"Rows       : {len(df):,}")
print(f"Columns    : {len(df.columns)}")
print(f"Customers  : {df['customer_id'].nunique():,}")
print(f"Restaurants: {df['restaurant_id'].nunique():,}")
print(f"File       : {OUTPUT_FILE}")

print("\nCuisine Distribution:")
print(
    df["cuisine"]
    .value_counts()
)

print("\nOrder Status:")
print(
    df["order_status"]
    .value_counts()
)

print("\nCity Distribution:")
print(
    df["city"]
    .value_counts()
)

print("\nSample Data:")
print(
    df.head()
)

print("\n" + "=" * 60)
print("              FOODLENS READY 🚀")
print("=" * 60)
