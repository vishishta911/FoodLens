# 🍽️ FoodLens AI

### AI-Powered Food Delivery Analytics & Customer Intelligence Platform

FoodLens AI is a data-driven food delivery analytics and recommendation platform that analyzes customer ordering behavior, restaurant performance, cuisine preferences, delivery patterns, and customer segments.

The system combines **data preprocessing, business analytics, machine learning-based customer segmentation, cuisine preference prediction, and personalized restaurant recommendations** into an interactive Streamlit dashboard.

---

## 🚀 Project Overview

Food delivery platforms generate large volumes of customer, restaurant, order, delivery, and transaction data.

FoodLens AI processes this data to answer questions such as:

- Which cuisines are most popular?
- Which cities generate the most revenue?
- What are the peak ordering periods?
- How does delivery time affect customer ratings?
- How can customers be segmented based on their behavior?
- What cuisine is a customer likely to prefer?
- Which restaurants should be recommended to a customer?

The platform provides both **business intelligence** and **AI-powered customer recommendations**.

---

## ✨ Key Features

### 📊 1. Executive Overview

Provides a high-level view of the food delivery business.

**Includes:**

- Total orders
- Delivered orders
- Revenue
- Average order value
- Cancellation rate
- Monthly revenue trends
- Cuisine distribution
- City performance

### 🏙️ 2. Business Analytics

Provides detailed business-level analysis.

**Features include:**

- City-wise performance
- Cuisine performance
- Revenue by cuisine
- Average order value
- Restaurant ratings
- Delivery performance
- Average delivery time
- Discount analysis
- Weather impact
- Restaurant performance
- Peak ordering hours
- Meal-period analysis
- Weekend vs weekday behavior

### 👥 3. Customer Segmentation

Customers are grouped according to their ordering behavior using **K-Means clustering**.

**Customer behavior features include:**

- Total orders
- Average order value
- Total spending
- Average rating
- Average delivery time
- Average discount
- Average distance
- Average quantity
- Average delivery fee

**The dashboard provides:**

- Customer cluster distribution
- Cluster profiles
- Customer lookup
- Customer value segment
- Predicted cuisine preference

### 🤖 4. Cuisine Preference Prediction

FoodLens AI uses a **Random Forest Classifier** to predict a customer's preferred cuisine.

#### Customer Behavior Features

- Total orders
- Average order value
- Total spending
- Average rating
- Average delivery time
- Average discount
- Average distance
- Average quantity
- Average delivery fee

#### Cuisine History Features

Historical order counts for each cuisine are also used as model features.

The trained model is saved as:

```text
models/preference_prediction_model.pkl
```

### 🍽️ 5. Personalized Restaurant Recommendation

After predicting a customer's preferred cuisine, FoodLens AI recommends restaurants serving that cuisine.

The recommendation score combines:

- Restaurant rating — **50%**
- Restaurant popularity/order count — **30%**
- Average order value — **20%**

The system produces ranked restaurant recommendations.

**Example:**

```text
#1 Restaurant R00893
   Rating: 4.62
   Average Order Value: ₹461

#2 Restaurant R00739
   Rating: 4.69
   Average Order Value: ₹605

#3 Restaurant R00238
   Rating: 4.62
   Average Order Value: ₹557
```

### 🔮 6. Predict & Recommend

The interactive prediction page allows a user to enter customer behavior details and receive:

```text
Customer Details
       ↓
Machine Learning Model
       ↓
Predicted Cuisine
       ↓
Restaurant Ranking
       ↓
Personalized Recommendations
```

The user can enter:

- Total orders
- Average order value
- Total spending
- Average rating
- Average delivery time
- Average discount
- Average distance
- Average quantity
- Average delivery fee
- Previous cuisine order counts

The system then predicts the customer's preferred cuisine and recommends suitable restaurants.

---

# 🏗️ Project Architecture

```text
                         FOODLENS AI
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Food Delivery Dataset  │
                  │      200,000 Orders    │
                  └────────────┬───────────┘
                               │
                               ▼
                       ┌─────────────────┐
                       │ Data Generation │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Preprocessing  │
                       └────────┬────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
          ┌──────────────────┐    ┌────────────────────┐
          │ Business         │    │ Customer Features  │
          │ Analytics        │    │ & Behavior         │
          └────────┬─────────┘    └──────────┬─────────┘
                   │                         │
                   │                ┌────────┴─────────┐
                   │                ▼                  ▼
                   │       ┌────────────────┐  ┌─────────────────┐
                   │       │ K-Means        │  │ Random Forest   │
                   │       │ Segmentation   │  │ Prediction      │
                   │       └────────────────┘  └────────┬────────┘
                   │                                     │
                   │                                     ▼
                   │                            ┌─────────────────┐
                   │                            │ Cuisine         │
                   │                            │ Prediction      │
                   │                            └────────┬────────┘
                   │                                     │
                   │                                     ▼
                   │                            ┌─────────────────┐
                   │                            │ Recommendation  │
                   │                            │ System          │
                   │                            └────────┬────────┘
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                           ┌────────────────────┐
                           │ Streamlit Dashboard│
                           └────────────────────┘
```

---

# 📁 Project Structure

```text
FoodLens/
│
├── data/
│   ├── raw/
│   │   └── food_delivery_orders.csv
│   │
│   └── processed/
│       ├── processed_orders.csv
│       ├── customer_features.csv
│       ├── customer_predictions.csv
│       ├── recommendations.csv
│       ├── cluster_profiles.csv
│       ├── monthly_performance.csv
│       ├── city_analysis.csv
│       ├── cuisine_analysis.csv
│       ├── hourly_analysis.csv
│       ├── meal_period_analysis.csv
│       ├── weekend_analysis.csv
│       ├── delivery_analysis.csv
│       ├── discount_analysis.csv
│       ├── weather_analysis.csv
│       ├── restaurant_analysis.csv
│       └── preference_feature_importance.csv
│
├── src/
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── analysis.py
│   ├── customer_segmentation.py
│   ├── preference_prediction.py
│   └── recommendation.py
│
├── models/
│   └── preference_prediction_model.pkl
│
├── dashboard/
│   └── app.py
│
├── reports/
│
├── requirements.txt
└── README.md
```

---

# 🛠️ Technology Stack

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- Random Forest
- K-Means Clustering

### Model Management

- Joblib

### Visualization

- Plotly
- Streamlit

### Dashboard

- Streamlit

### Data Storage

- CSV

---

# 📊 Dataset

FoodLens AI uses a food delivery order dataset containing approximately:

```text
200,000 orders
```

The dataset contains information related to:

- Customers
- Orders
- Restaurants
- Cuisine
- Cities
- Order value
- Discounts
- Delivery time
- Ratings
- Distance
- Quantity
- Weather
- Order status

---

# ⚙️ Data Preprocessing

The preprocessing pipeline performs:

1. Data loading
2. Missing-value checks
3. Duplicate checks
4. Date conversion
5. Time conversion
6. Date feature extraction
7. Meal-period classification
8. Weekend identification
9. Final amount calculation
10. Discount ratio calculation
11. Delivery categorization
12. Order-value categorization
13. Customer-level feature generation
14. Favorite cuisine identification
15. Customer frequency categorization
16. Customer value segmentation

Processed datasets are stored inside:

```text
data/processed/
```

---

# 🤖 Machine Learning Pipeline

## Customer Segmentation

K-Means clustering is used to identify groups of customers based on behavioral characteristics.

The segmentation helps identify different types of customers based on:

- Ordering frequency
- Spending behavior
- Order value
- Delivery preferences
- Ratings
- Discounts

---

## Cuisine Preference Prediction

A Random Forest classification model predicts the customer's preferred cuisine.

### Model

```text
RandomForestClassifier
n_estimators = 200
max_depth = 12
random_state = 42
```

The trained model is stored as:

```text
models/preference_prediction_model.pkl
```

---

## Restaurant Recommendation

Restaurants belonging to the predicted cuisine are ranked using a recommendation score.

### Recommendation Formula

```text
Recommendation Score =
    0.50 × Rating
  + 0.30 × Popularity
  + 0.20 × Price Preference
```

Where:

- **Rating** = average restaurant rating
- **Popularity** = normalized order count
- **Price Preference** = normalized preference for lower average order value

The top restaurants are returned as personalized recommendations.

---

# 📈 Business Analytics Generated

FoodLens AI generates analytical datasets for:

```text
Monthly Performance
City Analysis
Cuisine Analysis
Hourly Analysis
Meal Period Analysis
Weekend Analysis
Delivery Analysis
Discount Analysis
Weather Analysis
Restaurant Analysis
```

These datasets are stored in:

```text
data/processed/
```

---

# 🖥️ Running the Project

## 1. Clone or Open the Project

Open the **FoodLens** folder in VS Code.

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Generate the Dataset

```bash
python src/data_generation.py
```

## 4. Run Preprocessing

```bash
python src/preprocessing.py
```

## 5. Run Business Analytics

```bash
python src/analysis.py
```

## 6. Run Customer Segmentation

```bash
python src/customer_segmentation.py
```

## 7. Train Preference Prediction Model

```bash
python src/preference_prediction.py
```

## 8. Generate Recommendations

```bash
python src/recommendation.py
```

## 9. Launch the Dashboard

From the project root:

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser.

---

# 🖥️ Dashboard Modules

FoodLens AI contains the following dashboard sections:

```text
📊 Executive Overview
🏙️ Business Analytics
👥 Customer Segmentation
🤖 AI Recommendations
🔮 Predict & Recommend
```

---

# 🔮 Predict & Recommend Workflow

The prediction page allows a user to enter details for a new customer.

### Example

```text
Total Orders             : 10
Average Order Value      : ₹500
Total Spending           : ₹5,000
Average Rating           : 4.5
Average Delivery Time    : 35 minutes
Average Discount         : 10%
Average Distance         : 5 km
Average Quantity         : 2
Average Delivery Fee     : ₹40
```

The user can also provide previous cuisine order counts.

FoodLens AI then performs:

```text
Input Customer Data
        ↓
Random Forest Model
        ↓
Predicted Cuisine
        ↓
Restaurant Filtering
        ↓
Recommendation Scoring
        ↓
Top 5 Restaurants
```

---

# 🎯 Project Objectives

The main objectives of FoodLens AI are:

- Analyze food delivery business data
- Understand customer ordering behavior
- Identify customer segments
- Predict customer cuisine preferences
- Analyze restaurant performance
- Identify business trends
- Generate personalized restaurant recommendations
- Provide an interactive analytics dashboard

---

# 💡 Example Use Cases

## For Customers

- Discover restaurants matching their predicted preferences
- Receive personalized recommendations
- Find restaurants based on cuisine and price behavior

## For Restaurants

- Understand customer behavior
- Analyze cuisine demand
- Monitor ratings and delivery performance
- Understand revenue trends

## For Business Teams

- Analyze city performance
- Study discount patterns
- Identify peak ordering periods
- Compare restaurant performance
- Understand customer segments

---

# 🔐 Model and Data Files

### Generated Model

```text
models/preference_prediction_model.pkl
```

### Generated Prediction Data

```text
data/processed/customer_predictions.csv
```

### Generated Recommendation Data

```text
data/processed/recommendations.csv
```

---

# 📌 Project Highlights

- 📦 200,000 food delivery orders
- 👥 Customer-level behavioral analysis
- 🔵 K-Means customer segmentation
- 🤖 Random Forest cuisine prediction
- 🍽️ Personalized restaurant recommendation
- 📊 Business intelligence analytics
- 🖥️ Interactive Streamlit dashboard
- 🔮 New-customer prediction workflow

---

# 🚀 Future Enhancements

Possible future improvements include:

- Real-time recommendation updates
- Collaborative filtering
- Hybrid recommendation models
- Restaurant location-based recommendations
- Deep learning recommendation models
- Real-time order streaming
- Cloud deployment
- Database integration
- User authentication
- Restaurant-specific dashboards

---

# 👩‍💻 Project

## FoodLens AI

An AI-powered food delivery analytics and customer intelligence platform combining **business analytics, machine learning, customer segmentation, preference prediction, and personalized recommendations**.
