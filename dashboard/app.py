import os
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

# ============================================================
# FOODLENS - INTERACTIVE DASHBOARD
# ============================================================

st.set_page_config(
    page_title="FoodLens AI",
    page_icon="🍽️",
    layout="wide"
)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

processed_dir = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

orders = pd.read_csv(
    os.path.join(
        processed_dir,
        "processed_orders.csv"
    )
)

customers = pd.read_csv(
    os.path.join(
        processed_dir,
        "customer_predictions.csv"
    )
)

recommendations = pd.read_csv(
    os.path.join(
        processed_dir,
        "recommendations.csv"
    )
)

cluster_profiles = pd.read_csv(
    os.path.join(
        processed_dir,
        "cluster_profiles.csv"
    )
)
# ------------------------------------------------------------
# Load preference prediction model
# ------------------------------------------------------------

model_file = os.path.join(
    BASE_DIR,
    "models",
    "preference_prediction_model.pkl"
)

preference_model = joblib.load(model_file)

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.title("🍽️ FoodLens AI")

st.sidebar.write(
    "AI-powered food delivery analytics "
    "and customer intelligence platform."
)

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Executive Overview",
        "🏙️ Business Analytics",
        "👥 Customer Segmentation",
        "🤖 AI Recommendations",
        "🔮 Predict & Recommend"
    ]
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "📊 Executive Overview":

    st.title("🍽️ FoodLens AI")
    st.subheader(
        "Online Food Delivery Analytics & Customer Intelligence"
    )

    # KPIs

    total_orders = len(orders)

    delivered_orders = len(
        orders[
            orders["order_status"] == "Delivered"
        ]
    )

    total_revenue = orders["final_amount"].sum()

    average_order_value = orders[
        "final_amount"
    ].mean()

    cancellation_rate = (
        len(
            orders[
                orders["order_status"] == "Cancelled"
            ]
        )
        / total_orders
        * 100
    )

    average_delivery = orders[
        "delivery_time_min"
    ].mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    col2.metric(
        "Delivered",
        f"{delivered_orders:,}"
    )

    col3.metric(
        "Revenue",
        f"₹{total_revenue / 1e7:.2f} Cr"
    )

    col4.metric(
        "Average Order",
        f"₹{average_order_value:.0f}"
    )

    col5.metric(
        "Cancellation",
        f"{cancellation_rate:.2f}%"
    )

    st.divider()

    # Revenue by month

    st.subheader("📈 Monthly Revenue")

    monthly = (
        orders
        .groupby("month")
        .agg(
            revenue=("final_amount", "sum"),
            orders=("order_id", "count")
        )
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="month",
        y="revenue",
        markers=True,
        title="Monthly Revenue Trend"
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Revenue (₹)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Two charts

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🍛 Cuisine Distribution")

        cuisine = (
            orders["cuisine"]
            .value_counts()
            .reset_index()
        )

        cuisine.columns = [
            "cuisine",
            "orders"
        ]

        fig = px.bar(
            cuisine,
            x="cuisine",
            y="orders",
            title="Orders by Cuisine"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("🏙️ City Performance")

        city = (
            orders
            .groupby("city")
            .agg(
                revenue=("final_amount", "sum")
            )
            .reset_index()
            .sort_values(
                "revenue",
                ascending=False
            )
        )

        fig = px.bar(
            city,
            x="city",
            y="revenue",
            title="Revenue by City"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# BUSINESS ANALYTICS
# ============================================================

elif page == "🏙️ Business Analytics":

    st.title("🏙️ Business Analytics")

    # City filter

    selected_city = st.selectbox(
        "Select City",
        ["All"] + sorted(
            orders["city"].unique()
        )
    )

    filtered = orders.copy()

    if selected_city != "All":

        filtered = filtered[
            filtered["city"] == selected_city
        ]

    st.write(
        f"Showing {len(filtered):,} orders"
    )

    # Cuisine analysis

    st.subheader("🍛 Cuisine Performance")

    cuisine_analysis = (
        filtered
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
            )
        )
        .reset_index()
        .sort_values(
            "revenue",
            ascending=False
        )
    )

    st.dataframe(
        cuisine_analysis,
        use_container_width=True
    )

    fig = px.bar(
        cuisine_analysis,
        x="cuisine",
        y="revenue",
        title="Revenue by Cuisine"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Delivery analysis

    st.subheader("🚚 Delivery Performance")

    delivery = (
        filtered
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
            )
        )
        .reset_index()
    )

    st.dataframe(
        delivery,
        use_container_width=True
    )

    fig = px.bar(
        delivery,
        x="delivery_category",
        y="average_delivery_time",
        title="Average Delivery Time"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Discount analysis

    st.subheader("🏷️ Discount Analysis")

    discount = (
        filtered
        .groupby("discount_percent")
        .agg(
            orders=("order_id", "count"),
            revenue=("final_amount", "sum"),
            average_order_value=(
                "final_amount",
                "mean"
            )
        )
        .reset_index()
    )

    st.dataframe(
        discount,
        use_container_width=True
    )


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

elif page == "👥 Customer Segmentation":

    st.title("👥 Customer Segmentation")

    st.subheader(
        "K-Means Customer Behavior Analysis"
    )

    # Cluster distribution

    if "cluster" in customers.columns:

        cluster_counts = (
            customers["cluster"]
            .value_counts()
            .reset_index()
        )

        cluster_counts.columns = [
            "cluster",
            "customers"
        ]

        fig = px.pie(
            cluster_counts,
            names="cluster",
            values="customers",
            title="Customer Cluster Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("📊 Cluster Profiles")

    st.dataframe(
        cluster_profiles,
        use_container_width=True
    )

    # Customer lookup

    st.subheader("🔎 Customer Lookup")

    customer_id = st.selectbox(
        "Select Customer",
        sorted(
            customers["customer_id"].unique()
        )
    )

    customer = customers[
        customers["customer_id"] == customer_id
    ].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Orders",
        int(customer["total_orders"])
    )

    col2.metric(
        "Average Order",
        f"₹{customer['average_order_value']:.0f}"
    )

    col3.metric(
        "Total Spending",
        f"₹{customer['total_spending']:.0f}"
    )

    col4.metric(
        "Favorite Cuisine",
        customer["favorite_cuisine"]
    )

    st.write(
        f"**Predicted Cuisine:** "
        f"{customer['predicted_cuisine']}"
    )

    if "customer_value_segment" in customer:

        st.write(
            f"**Customer Value Segment:** "
            f"{customer['customer_value_segment']}"
        )


# ============================================================
# AI RECOMMENDATIONS
# ============================================================

elif page == "🤖 AI Recommendations":

    st.title("🤖 AI-Powered Recommendations")

    st.write(
        "Personalized restaurant recommendations "
        "based on customer behavior and predicted cuisine preference."
    )

    customer_id = st.selectbox(
        "Select Customer",
        sorted(
            customers["customer_id"].unique()
        )
    )

    customer = customers[
        customers["customer_id"] == customer_id
    ].iloc[0]

    predicted_cuisine = customer[
        "predicted_cuisine"
    ]

    segment = customer.get(
        "customer_value_segment",
        "Regular Customer"
    )

    st.success(
        f"Predicted Preference: 🍛 {predicted_cuisine}"
    )

    st.info(
        f"Customer Segment: {segment}"
    )

    customer_recommendations = recommendations[
        recommendations["customer_id"]
        == customer_id
    ].sort_values(
        "rank"
    )

    st.subheader(
        "🍽️ Recommended Restaurants"
    )

    for _, restaurant in customer_recommendations.iterrows():

        col1, col2, col3, col4 = st.columns(4)

        col1.write(
            f"### #{int(restaurant['rank'])}"
        )

        col2.write(
            f"**{restaurant['restaurant_id']}**"
        )

        col3.write(
            f"⭐ {restaurant['restaurant_rating']:.2f}"
        )

        col4.write(
            f"₹{restaurant['average_order_value']:.0f}"
        )

        st.divider()

    st.caption(
        "Recommendations combine cuisine preference, "
        "restaurant rating, popularity and average order value."
    )
    
# ============================================================
# PREDICT & RECOMMEND
# ============================================================

elif page == "🔮 Predict & Recommend":

    st.title("🔮 Predict & Recommend")

    st.write(
        "Enter customer behavior details to predict "
        "their preferred cuisine and get personalized "
        "restaurant recommendations."
    )

    st.divider()

    # --------------------------------------------------------
    # Customer behavior
    # --------------------------------------------------------

    st.subheader("👤 Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:

        total_orders = st.number_input(
            "Total Orders",
            min_value=1,
            max_value=500,
            value=10
        )

        average_order_value = st.number_input(
            "Average Order Value (₹)",
            min_value=50.0,
            max_value=5000.0,
            value=500.0
        )

        total_spending = st.number_input(
            "Total Spending (₹)",
            min_value=50.0,
            max_value=100000.0,
            value=5000.0
        )

    with col2:

        average_rating = st.number_input(
            "Average Rating",
            min_value=1.0,
            max_value=5.0,
            value=4.5,
            step=0.1
        )

        average_delivery_time = st.number_input(
            "Average Delivery Time (min)",
            min_value=5.0,
            max_value=120.0,
            value=35.0
        )

        average_discount = st.number_input(
            "Average Discount (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0
        )

    with col3:

        average_distance = st.number_input(
            "Average Distance (km)",
            min_value=0.1,
            max_value=50.0,
            value=5.0
        )

        average_quantity = st.number_input(
            "Average Quantity",
            min_value=1.0,
            max_value=20.0,
            value=2.0
        )

        average_delivery_fee = st.number_input(
            "Average Delivery Fee (₹)",
            min_value=0.0,
            max_value=500.0,
            value=40.0
        )

    # --------------------------------------------------------
    # Cuisine history
    # --------------------------------------------------------

    st.subheader("🍛 Previous Cuisine Orders")

    st.caption(
        "Enter approximately how many times the customer "
        "has previously ordered each cuisine."
    )

    cuisines = sorted(
        orders["cuisine"].dropna().unique()
    )

    cuisine_counts = {}

    cols = st.columns(3)

    for i, cuisine in enumerate(cuisines):

        with cols[i % 3]:

            cuisine_counts[cuisine] = st.number_input(
                cuisine,
                min_value=0,
                max_value=500,
                value=0,
                step=1,
                key=f"prediction_{cuisine}"
            )

    st.divider()

    # --------------------------------------------------------
    # Predict button
    # --------------------------------------------------------

    if st.button(
        "🚀 Predict Preference & Recommend",
        type="primary"
    ):

        # Behavior features

        input_data = {
            "total_orders": total_orders,
            "average_order_value": average_order_value,
            "total_spending": total_spending,
            "average_rating": average_rating,
            "average_delivery_time": average_delivery_time,
            "average_discount": average_discount,
            "average_distance": average_distance,
            "average_quantity": average_quantity,
            "average_delivery_fee": average_delivery_fee
        }

        # Cuisine history features

        for cuisine in cuisines:

            feature_name = (
                "orders_" +
                cuisine.lower().replace(" ", "_")
            )

            input_data[feature_name] = (
                cuisine_counts[cuisine]
            )

        # ----------------------------------------------------
        # Arrange features in exact model order
        # ----------------------------------------------------

        input_df = pd.DataFrame([input_data])

        try:

            input_df = input_df[
                preference_model.feature_names_in_
            ]

        except AttributeError:

            st.error(
                "Model feature information is unavailable."
            )
            st.stop()

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predicted_cuisine = (
            preference_model.predict(input_df)[0]
        )

        # Prediction probability

        probabilities = (
            preference_model.predict_proba(input_df)[0]
        )

        max_probability = probabilities.max()

        confidence = max_probability * 100

        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        st.success(
            f"🍛 Predicted Cuisine: **{predicted_cuisine}**"
        )

        st.info(
            f"Model confidence: **{confidence:.1f}%**"
        )

        # ----------------------------------------------------
        # Restaurant recommendations
        # ----------------------------------------------------

        st.subheader(
            "🍽️ Recommended Restaurants"
        )

        restaurant_data = (
            orders[
                orders["order_status"] == "Delivered"
            ]
            .groupby(
                ["restaurant_id", "cuisine", "city"]
            )
            .agg(
                order_count=(
                    "order_id",
                    "count"
                ),

                average_rating=(
                    "rating",
                    "mean"
                ),

                average_order_value=(
                    "order_value",
                    "mean"
                )
            )
            .reset_index()
        )

        recommendations = restaurant_data[
            restaurant_data["cuisine"] ==
            predicted_cuisine
        ].copy()

        if recommendations.empty:

            st.warning(
                "No restaurants found for the "
                "predicted cuisine."
            )

        else:

            # Same recommendation logic as your
            # existing recommendation system

            max_orders = recommendations[
                "order_count"
            ].max()

            max_order_value = recommendations[
                "average_order_value"
            ].max()

            if max_orders == 0:
                max_orders = 1

            if max_order_value == 0:
                max_order_value = 1

            recommendations["score"] = (

                recommendations[
                    "average_rating"
                ] * 0.50

                +

                recommendations[
                    "order_count"
                ] / max_orders * 0.30

                +

                (
                    1 -

                    recommendations[
                        "average_order_value"
                    ] / max_order_value

                ) * 0.20
            )

            recommendations = (
                recommendations
                .sort_values(
                    "score",
                    ascending=False
                )
                .head(5)
            )

            for rank, (_, restaurant) in enumerate(
                recommendations.iterrows(),
                start=1
            ):

                col1, col2, col3, col4, col5 = (
                    st.columns(5)
                )

                col1.write(
                    f"### #{rank}"
                )

                col2.write(
                    f"**{restaurant['restaurant_id']}**"
                )

                col3.write(
                    f"🍛 {restaurant['cuisine']}"
                )

                col4.write(
                    f"⭐ {restaurant['average_rating']:.2f}"
                )

                col5.write(
                    f"₹{restaurant['average_order_value']:.0f}"
                )

                st.caption(
                    f"📍 {restaurant['city']}  |  "
                    f"Recommendation Score: "
                    f"{restaurant['score']:.3f}"
                )

                st.divider()