# DSN Mart Retail Sales Prediction

This repository contains the end-to-end machine learning solution for the **Data Science Nigeria (DSN) AI Bootcamp Qualification Hackathon**. The objective is to forecast product sales (`total_sales`) across various DSN Mart retail locations to support inventory optimization and supply chain management.

---

## 📌 Project Overview & Pipeline

The project follows a structured data science methodology:

$$\text{Understand} \longrightarrow \text{Analyse} \longrightarrow \text{Feature Engineer} \longrightarrow \text{Model} \longrightarrow \text{Evaluate} \longrightarrow \text{Predict}$$

* **Problem Formulation:** Supervised tabular regression.
* **Evaluation Metric:** Root Mean Squared Error (RMSE).
* **Validation Strategy:** 5-Fold Cross-Validation (`KFold(n_splits=5, shuffle=True)`).

---

## 📊 Exploratory Data Analysis & Key Insights

1. **Target Distribution (`total_sales`):**
   * The raw sales target exhibits a positive skewness of `+1.15`, driven by high-value sales transactions.
   * Applying a log transform (`np.log1p`) overcorrected the target into a left-skewed distribution (`-0.90`), penalizing the RMSE metric. Consequently, modeling was prioritized directly on the original scale to align with the squared-error objective.
2. **Missing Value Treatment:**
   * Handled missing records in `product_weight_kg` and `store_size` using categorical/median grouping strategies and missingness indicator flags.
   * Cleaned zero-value entries in `shelf_visibility` by imputing category-level means.

---

## 🛠️ Feature Engineering

Key features engineered to boost gradient boosting performance include:
* **Relative Metrics:**
  * `item_visibility_ratio_store`: Ratio of an item's shelf prominence relative to the store average.
  * `price_to_cat_mean`: Ratio of product price relative to category-level average price.
  * `price_per_kg`: Price per unit weight.
* **Categorical Interactions:**
  * Combined structural store features (e.g., `store_size_format`, `tier_format`).
  * Frequency/count encoding across key categorical variables (`product_category`, `store_code`, `store_format`).
* **Non-linear Signals:**
  * Interaction terms such as visual product exposure (`visibility * price`).
  * Within-store percentile ranking of product prices.

---

## 🤖 Modeling & Validation

* **Primary Model:** Gradient Boosting (CatBoost / LightGBM Regressor).
* **Handling Categoricals:** Leveraged native categorical feature handling with appropriate string imputation for missing levels.
* **Regularization & Early Stopping:** Evaluated validation RMSE per fold using early stopping (100 rounds) to prevent overfitting to the public leaderboard.
* **Inference:** Test set predictions were computed as the ensemble average across all 5 trained cross-validation fold models.

---

## 📁 Repository Structure

```text
├── dsn_prediction_price.ipynb   # Complete Jupyter Notebook (EDA, FE, CV, Training)
├── submission.csv               # Final generated test predictions file
└── README.md                    # Project documentation & summary
