## 🎯 Project Overview

**Problem:** A real-estate firm wants to estimate the price of a house from its characteristics (area, bedrooms, location quality, age and so on) so it can price listings and judge investments with data instead of guesswork.

**Goal:** Build and compare several regression models, understand *how* and *why* their accuracy differs, and recommend the one that best balances accuracy and reliability.

**What makes this project different from a "just fit a model" notebook:**
- ✅ Every regression **assumption is tested** with plots, not assumed.
- ✅ **Gradient descent is coded from scratch** (Batch, Stochastic and Mini-Batch) and compared on convergence and speed.
- ✅ **Overfitting and underfitting are demonstrated with evidence**, using train vs test error.
- ✅ Results are translated into **plain business language**.

---

## 🏆 Headline Results

| Model | Features used | MAE | RMSE | Test R² | Verdict |
|---|---|---|---|---|---|
| Simple Linear Regression | Area only | ₹62.9 lakh | ₹81.8 lakh | **0.563** | ❌ Underfits |
| Multiple Linear Regression | All features | ₹26.1 lakh | ₹35.5 lakh | **0.918** | ✅ Strong |
| **Polynomial Regression (degree 2)** | All features | **₹16.6 lakh** | **₹22.8 lakh** | **0.966** | 🏆 **Best** |
| Polynomial Regression (degree 5) | All features | — | ₹105.4 lakh | 0.274 | ❌ Overfits badly |

> **In one line:** adding more information (more features) cut the average error by more than half, and a gentle curve (degree 2) improved it further, but over-complicating the curve (degree 5) destroyed performance.

*(1 lakh = 100,000 INR, 1 crore = 10,000,000 INR. The average house in the test set costs about ₹2.4 crore.)*

---

## 📂 Dataset

**File:** `House_Prediction_Data.csv`: **4,200 houses**, **12 columns**, no missing values.

| Column | Type | Description |
|---|---|---|
| `house_id` | ID | Unique row label (**not used as a feature**) |
| `area_sqft` | Numeric | Built-up area in square feet (450 to 4,202) |
| `bedrooms` | Numeric | Number of bedrooms (1 to 7) |
| `bathrooms` | Numeric | Number of bathrooms (1 to 6) |
| `location_score` | Numeric | Quality of the location, 1 (poor) to 10 (excellent) |
| `age_years` | Numeric | Age of the house in years (1 to 80) |
| `distance_city_km` | Numeric | Distance from the city centre in km |
| `lot_size_sqft` | Numeric | Size of the plot in square feet |
| `has_garage` | Binary | 1 = has a garage, 0 = no garage |
| `has_pool` | Binary | 1 = has a pool, 0 = no pool |
| `renovation_years_ago` | Numeric | Years since the last renovation (0 = recently renovated) |
| **`house_price_inr`** | **Target** | **Sale price in INR (₹8 lakh to ₹7.6 crore)** |

**Train/test split:** 80% training (3,360 rows) and 20% testing (840 rows), using `random_state=42` so results are reproducible.

---

## 🔄 Project Workflow

```
 Data  ──►  Explore  ──►  Simple LR  ──►  Check       ──►  Metrics
 (CSV)      & plot        (area only)     assumptions       (MAE/RMSE/R²)
                                                               │
 Report  ◄──  Bias–Variance  ◄──  Gradient  ◄──  Polynomial  ◄──  Multiple LR
 (Part I)     (Part H)            Descent (G)     (Part F)         (Part E)
```

The notebook is organised into nine parts that follow the assignment:

| Part | Topic | What it does |
|---|---|---|
| **A** | Theory | Supervised learning, regression vs classification, assumptions, bias–variance, over/underfitting |
| **B** | Data preparation | Identify X and y, visualise every feature against price, train/test split |
| **C** | Simple Linear Regression | Fit price vs area, plot the line, interpret slope and intercept, validate assumptions |
| **D** | Evaluation metrics | MSE, MAE, RMSE, R², Adjusted R² and what each reveals |
| **E** | Multiple Linear Regression | Use all features and explain the improvement |
| **F** | Polynomial Regression | Degrees 1 to 5, linear vs polynomial comparison, over/underfitting detection |
| **G** | Gradient Descent | Batch, SGD and Mini-Batch implemented from scratch and compared |
| **H** | Bias–Variance | Train vs test error across models and the effect of complexity |
| **I** | Final report | Summary of best model, optimisation, fit quality and business meaning |

---

## 🔍 Detailed Findings

### 1. Simple Linear Regression (Price vs Area)
- **Slope ≈ ₹14,788 per sq ft:** every extra square foot adds about ₹14,800 to the predicted price.
- **Intercept ≈ −₹11.6 lakh:** the predicted price at 0 sq ft. It has no real-world meaning (the smallest house is 450 sq ft) and only positions the line.
- **R² ≈ 0.56:** area alone explains about 56% of the price variation.

### 2. Regression assumptions (tested on the simple model)

| Assumption | Check used | Result |
|---|---|---|
| Linearity | Scatter plot with fitted line | ✅ Satisfied (correlation ≈ 0.75) |
| Independence of errors | Durbin-Watson statistic | ✅ ≈ 1.99 (ideal is about 2) |
| Normality of residuals | Histogram and Q-Q plot | ✅ Mostly satisfied |
| Constant variance (homoscedasticity) | Residuals vs predicted plot | ⚠️ **Violated**: errors grow for expensive houses (a "funnel" shape) |

### 3. Why Multiple Regression is so much better
R² jumped from **0.56 to 0.92** and the average error fell from ₹63 lakh to ₹26 lakh. The reason is that price depends on many factors, and the biggest missing piece was **location score** (about ₹30.7 lakh per point), which is almost uncorrelated with area, so area alone could never capture it.

### 4. Polynomial Regression: finding the sweet spot

| Degree | Train R² | Test R² | Test RMSE | Observation |
|---|---|---|---|---|
| 1 | 0.9247 | 0.9178 | ₹35.5 lakh | Slightly too simple |
| **2** | **0.9685** | **0.9664** | **₹22.7 lakh** | **Best fit** |
| 3 | 0.9712 | 0.9646 | ₹23.3 lakh | Same as degree 2 |
| 4 | 0.9781 | 0.9271 | ₹33.4 lakh | Overfitting begins |
| 5 | 0.9895 | 0.2744 | ₹105.4 lakh | Severe overfitting |

**The classic overfitting signature:** training R² keeps climbing (0.97 → 0.99) while test R² collapses (0.97 → 0.27). The model memorises noise instead of learning the pattern.

### 5. Gradient Descent: three ways to reach the same answer
All three methods were written from scratch and reached the **same error as the closed-form solution** (MSE ≈ 6.6 × 10¹³), proving they find the true best-fit line.

| Method | Rows per update | Convergence | Training time (this dataset) |
|---|---|---|---|
| **Batch GD** | All rows | Smooth but slow: about 12 epochs | ⚡ Fastest (vectorised) |
| **Stochastic GD** | 1 row | Very fast per epoch (about 1), but noisy | 🐢 Slowest (one Python update per row) |
| **Mini-Batch GD** | 32 rows | Fast and fairly smooth | ⚖️ In between |

**Takeaway:** Mini-Batch GD offers the best practical balance, which is why it is the standard choice for large datasets. Feature scaling (`StandardScaler`) was essential for convergence.

### 6. Bias–Variance analysis

| Model | Train RMSE | Test RMSE | Gap | Diagnosis |
|---|---|---|---|---|
| Simple Linear | ₹81.0 lakh | ₹81.8 lakh | ₹0.8 lakh | **High bias** (underfitting) |
| Multiple Linear | ₹34.0 lakh | ₹35.5 lakh | ₹1.5 lakh | Low bias, low variance |
| **Polynomial (deg 2)** | **₹22.0 lakh** | **₹22.7 lakh** | **₹0.7 lakh** | ✅ **Best balance** |
| Polynomial (deg 5) | ₹12.7 lakh | ₹105.4 lakh | ₹92.7 lakh | **High variance** (overfitting) |

As model complexity increases, **bias falls but variance rises**. Training error always drops, while test error follows a U-shape. The best model sits at the bottom of the U.

---

## 💼 Business Interpretation

- **Accuracy:** the best model estimates a house price with a typical error of about **₹23 lakh**, which is **under 10%** of the average price (about ₹2.4 crore).
- **What drives price most:**
  1. 📍 **Location score:** about **+₹30.7 lakh per point** (the strongest driver)
  2. 📐 **Area:** about **+₹13,800 per sq ft**
  3. 🏊 **Pool, bedrooms and bathrooms** add smaller amounts
  4. 📉 **Older houses and houses further from the city** are cheaper
- **How a firm can use it:** setting listing prices, valuing properties, spotting under- or over-priced houses, and guiding investment decisions.

---

## 🗂️ Repository Structure

```
house-price-prediction/
│
├── Predictive_Insight_Engine.ipynb   # Main notebook (Parts A to I)
├── House_Prediction_Data.csv         # Dataset (4,200 houses)
├── README.md                         # You are here
├── requirements.txt                  # Python dependencies (optional)
└── images/                           # Screenshots of key plots (optional)
```

---

## 🛠️ Tech Stack

| Purpose | Library |
|---|---|
| Data handling | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Modelling | `scikit-learn` (LinearRegression, PolynomialFeatures, SGDRegressor, StandardScaler, pipelines) |
| Statistics | `scipy` (Q-Q plots, normality checks) |
| Environment | Jupyter Notebook |

---

## 🧠 Key Learnings

1. **More relevant information beats more complexity:** adding the right features raised R² from 0.56 to 0.92.
2. **Complexity has a sweet spot:** a gentle curve (degree 2) helped, but a high-degree curve destroyed performance on unseen data.
3. **Always compare train vs test error:** that gap is the quickest way to detect overfitting.
4. **Gradient descent is a path, not a different answer:** Batch, SGD and Mini-Batch all land on the same solution but differ in speed and stability.
5. **Check assumptions, don't assume them:** the residual plot revealed a funnel pattern (heteroscedasticity) that a metric alone would never show.
6. **Scale your features:** gradient descent does not converge reliably without it.

---

## ⚠️ Limitations & Future Work

**Limitations**
- Errors grow for expensive houses (heteroscedasticity), so predictions for luxury homes are less reliable.
- The model only knows these 11 features. Real prices also depend on market conditions, neighbourhood trends and house condition.
- Results come from a single 80/20 split.

**Possible improvements**
- 📈 Log-transform the price to stabilise the error spread.
- 🔁 Use **k-fold cross-validation** for more reliable scores.
- 🧮 Add **Ridge / Lasso** regularisation to control overfitting in high-degree models.
- 🌲 Try **Random Forest** or **Gradient Boosting** for non-linear patterns.
- 🌐 Deploy as a small **Streamlit** app where users enter house details and get an instant price estimate.

---

## 🎬 Demo Video

📺 **Walkthrough video:** [Add your YouTube / Google Drive link here](#)

---

## 👤 Author

**Bhatti Pushkar**
Data Analysis Project — Red & White Skill Education

⭐ If you found this project helpful, consider giving the repository a star!
