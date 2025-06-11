# Titanic Survival Prediction 🚢

This project is a complete machine learning pipeline built to predict survival on the Titanic dataset. It leverages various classification algorithms and their hyperparameter optimization techniques to compare performance and interpret outcomes. 

## 📌 Objective

The primary objective is not just to predict survival, but to **maximize the recall** — minimizing false negatives (i.e., predicting someone didn't survive when they actually did). In a real-world disaster scenario, it's far more costly to miss saving a life than to raise a false alarm.

## 🛠️ Tools & Libraries Used

- **Languages**: Python
- **Libraries**: 
  - pandas, numpy
  - matplotlib, seaborn
  - scikit-learn
  - xgboost
  - lightgbm

## 🔍 Problem Statement

Given features like age, gender, class, fare, etc., predict whether a passenger survived the Titanic shipwreck.

---

## 🔄 Workflow Overview

1. **Data Preprocessing**
   - Handling missing values (e.g., imputing `Age`)
   - Encoding categorical features (`Sex`, `Embarked`)
   - Feature scaling using `StandardScaler`

2. **Model Building** (Baseline)
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - LightGBM
   - K-Nearest Neighbors
   - Support Vector Machine
   - Naive Bayes

3. **Hyperparameter Tuning**
   - `GridSearchCV` and `RandomizedSearchCV`
   - Evaluated using metrics like Accuracy, Recall, Precision, F1 Score, and ROC AUC

4. **Model Evaluation**
   - Focused on **Recall Score** due to the nature of the problem
   - Compared each model before and after tuning

---

## 🎯 Key Learnings

- Importance of trade-offs in metrics: **High accuracy doesn’t guarantee high recall.**
- Hyperparameter tuning can improve model performance but may reduce recall.
- Some models perform better than others for imbalanced or life-critical datasets.
- The role of probabilistic outputs in calculating ROC AUC.

---

## 📊 Evaluation Metrics

Each model was evaluated using:
- **Accuracy**
- **Precision**
- **Recall** (Primary Metric)
- **F1 Score**
- **ROC AUC Score**

---

## 🧠 Conclusion

In scenarios like survival prediction, **recall is more important than just accuracy**. This project taught the balance between:
- Predictive performance
- Algorithm choice
- Impact of hyperparameters
- Interpreting real-world cost of false negatives

---

## 🚀 Future Scope

- Feature engineering from names, ticket, or cabin info
- Use of ensemble models like stacking
- Incorporate SHAP or LIME for model explainability

## Screenshot
---![Model_Comparison](ModelComparison.png)

