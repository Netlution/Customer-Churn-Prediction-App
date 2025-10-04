# 📊 Customer Churn & Value Prediction System

A **Machine Learning project** built using **Streamlit**, **Scikit-learn**, and **Pandas**, designed to predict:

1. **Customer Churn (Classification)** — Whether a customer will leave or stay.  
2. **Customer Value (Regression)** — Estimate the continuous value of a customer based on their subscription and usage data.

---

## 🚀 Project Overview

Customer churn is one of the biggest challenges in the telecom industry. Retaining high-value customers requires the ability to **predict customer behavior** and **estimate their value**.

This project provides two Streamlit-based modules:

- **Customer Churn Prediction (Classification)** — Predicts if a customer will **stay (0)** or **leave (1)**.  
- **Customer Value Regression (Continuous Prediction)** — Estimates the **monetary value** of each customer.

🔗 **Live Demo:** [Streamlit App](https://customa-churn-prediction-model.streamlit.app/)

---

## 🧠 Dataset Information

This dataset is randomly collected from an **Iranian telecom company’s database** over a **12-month period**.  
It contains **3,150 rows** (customers) and **13 columns** (features).

| Feature | Description |
|----------|--------------|
| Call Failure | Number of failed calls |
| Complains | Binary (1 if complaint lodged, else 0) |
| Subscription Length | Duration of subscription (months) |
| Charge Amount | Total amount charged to the customer |
| Seconds of Use | Total call duration (in seconds) |
| Frequency of Use | Frequency of call usage |
| Frequency of SMS | Number of SMS sent |
| Distinct Called Numbers | Unique contacts called |
| Age Group | Customer age group |
| Tariff Plan | 0 or 1 — type of tariff plan |
| Status | Service status (1 or 2) |
| Customer Value | Continuous numerical value (target for regression) |
| Churn | 1 if customer left, 0 if retained (target for classification) |

> ⚙️ **Note:** All attributes except *churn* represent aggregated data from the first 9 months.  
> The *churn* label shows customer state at the end of 12 months, with a 3-month planning gap.

---

## 🧩 Machine Learning Models

| Task | Model Used | File |
|------|-------------|------|
| Regression (Customer Value) | Random Forest Regressor | `rf_regression_model.pkl` |
| Classification (Customer Churn) | Random Forest Classifier | `rf_churn_model.pkl` |

Models were trained in the Jupyter notebook:  
`trainedmodel.ipynb`

---

## 💻 Streamlit Applications

### 🅰️ Customer Churn Prediction (Classification)
Predicts if a customer will **churn (1)** or **stay (0)** based on key behavioral metrics.

**Input Features:**
- Call Failure  
- Complains  
- Subscription Length  
- Charge Amount  
- Seconds of Use  
- Frequency of Use  
- Frequency of SMS  
- Distinct Called Numbers  
- Age Group  
- Tariff Plan  
- Status  
- Customer Value  

**Output:**
- ✅ Customer will stay  
- ❌ Customer will leave  

---

### 📈 Customer Value Prediction (Regression)
Predicts the **continuous customer value** (e.g., revenue contribution) using regression analysis.

**Input Features:**
- Call Failures  
- Complains  
- Subscription Length  
- Charge Amount  
- Seconds of Use  
- Frequency of Calls & SMS  
- Distinct Called Numbers  
- Age Group  
- Tariff Plan & Status  
- Churn  

**Output:**
- 💰 Estimated Customer Value

---

## 🛠️ Installation and Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
