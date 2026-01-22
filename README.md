# Titanic Survival Prediction using Random Forest
# Project Overview

This project focuses on building a **Random Forest Classifier** to predict passenger survival on the **Titanic dataset**. Unlike a minimal or pre-cleaned workflow, this project intentionally includes **extra EDA and data cleaning steps** to better understand the data and improve model robustness.

The final model also provides **feature importance analysis**, which helps explain *why* the model makes certain predictions.

** Dataset**

* **Source:** ("titanic.csv")`, it's basically intentionaly changed to perform and show some step
* **Target Variable:** `survived`

  * `0` → Did not survive
  * `1` → Survived

### Selected Features

| Feature  | Description                       |
| -------- | --------------------------------- |
| pclass   | Passenger class (1st, 2nd, 3rd)   |
| sex      | Gender of passenger               |
| age      | Age (continuous)                  |
| sibsp    | Number of siblings/spouses aboard |
| parch    | Number of parents/children aboard |
| fare     | Ticket fare                       |
| embarked | Port of embarkation               |

---

## Exploratory Data Analysis (EDA)

Key EDA steps performed:

* Checked missing values (`age`, `embarked`)
* Analyzed survival distribution across:

  * Gender
  * Passenger class
  * Age groups
* Outlier inspection for `fare`
* Correlation analysis

These steps helped identify:

* Strong survival dependency on **sex** and **fare**
* Mild influence of family-related features (`sibsp`, `parch`)

---

##  Data Cleaning & Feature Engineering

The following transformations were applied:

### 1. Missing Value Treatment

* `age` → filled using **median**
* `embarked` → filled using **mode**

### 2. Encoding

* `sex` → one-hot encoding
* `embarked` → one-hot encoding

### 3. Scaling

* Applied **StandardScaler** to all numerical features
 *Note:* Scaling is **not required** for Random Forest but was included to maintain a consistent ML pipeline and for comparison with other models

---

## Model Used

**Random Forest Classifier**

### Why Random Forest?

* Handles non-linear relationships
* Robust to outliers
* Built-in feature importance
* Performs well without heavy parameter tuning

### Key Parameters

* `n_estimators = 200`
* `random_state = 42`

---

## Model Evaluation

Evaluation was performed on a held-out test set (80/20 split):

* **Accuracy:** ~80–83%
* **Metrics Used:**

  * Accuracy Score
  * Confusion Matrix
  * Classification Report (Precision, Recall, F1-score)

---

##  Feature Importance Analysis

The trained Random Forest model provides feature importance scores.

### Top Important Features (Observed)

1. **Fare**
2. **Age**
3. **Sex (male/female)**
4. **Pclass**

Less influential features include:

* SibSp
* Parch
* Embarked categories

** A bar plot was used to visualize these importance scores for better interpretability.**


## Key Learnings

* Proper EDA and cleaning significantly improve model reliability
* Tree-based models do not require scaling, but scaling does not harm performance
* Feature importance helps explain real-world behavior (e.g., higher fare → better survival odds)
---

##  Tech Stack

* Python
* Pandas, NumPy
* Seaborn, Matplotlib
* Scikit-learn

---

##  Conclusion

This project demonstrates a **realistic ML workflow**, including EDA, cleaning, scaling, modeling, and interpretation. It goes beyond a toy example and closely resembles how applied machine learning projects are structured in practice.
