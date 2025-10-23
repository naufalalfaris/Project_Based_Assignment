# Energy Efficiency — Supervised Regression Project

**Predicting Heating & Cooling Load using machine learning**

**Project type:** Regression · Supervised Learning · Model Tuning  
**Team:** Fauzan Naufal Rizqi, Naufal Alfarisi, Leonardus Adi Widjayanto  
**Dataset:** Energy Efficiency (UCI Machine Learning Repository)

---

## 🔍 Project Overview
This project aims to predict the **Heating Load (Y1)** and **Cooling Load (Y2)** of building configurations using the Energy Efficiency dataset. The dataset contains building attributes (e.g., relative compactness, surface area, wall area, roof area, overall height, glazing area, orientation) and the corresponding energy loads. The objective is to build robust regression models that generalize well and provide actionable insight for energy optimization.

---

## 🧩 Dataset
- **Source:** UCI Machine Learning Repository  
- **Samples:** Several hundred observations across building configurations  
- **Features:** Structural and geometric features derived from building designs  
- **Targets:** `Heating Load` and `Cooling Load` (continuous values)

---

## ⚙️ Data Preprocessing
- Data loading and exploratory analysis (distribution, correlation)  
- Data cleaning and type checks  
- Train/test split (train_test_split)  
- Feature scaling: `StandardScaler`, (also tested `MinMaxScaler`)  
- Residual and error analysis to understand model behavior

---

## 🧠 Models & Methodology
Several regression algorithms were implemented and compared:
- **Support Vector Regression (SVR)** — hyperparameter tuning via `GridSearchCV` (multiple kernel schemes)  
- **Random Forest Regressor**  
- **Gradient Boosting Regressor**  
- **MLPRegressor (ANN)**

Model selection was based on **MSE, MAE, and R²**. Grid search tuning was applied especially for SVR to find the best combination of kernel, C, gamma, and epsilon.

---

## 📈 Evaluation (Best reported results)
> Results shown are from the tuned SVM scheme (best performing configuration in notebook)

- **Heating Load (Y1)**  
  - MSE: **0.8641**  
  - MAE: **0.7017**  
  - R²: **0.9917**

- **Cooling Load (Y2)**  
  - MSE: **3.5180**  
  - MAE: **1.2599**  
  - R²: **0.9619**

Both targets show high R², indicating strong predictive performance from the tuned models.

---

## 💡 Key Insights
- Geometric and material features of a building strongly influence predicted energy loads.  
- Tuned SVR yields excellent performance for both heating and cooling predictions.  
- Residual analysis suggests low bias and small error variance for tuned models.

---

## 🚀 Future Improvements
- Add **feature importance** and interpretability (e.g., SHAP values) to explain which features drive predictions.  
- Use **cross-validation** metrics to report model stability (k-fold CV RMSE/R²).  
- Experiment with **ensemble stacking** or advanced learners (XGBoost / LightGBM) for further gains.  
- Wrap the best model into a small **API** (Flask/FastAPI) + demo UI for practical deployment.

---

## 🧰 Tools & Libraries
- Python, pandas, numpy, matplotlib, seaborn  
- Scikit-learn (SVR, RandomForestRegressor, GradientBoostingRegressor, MLPRegressor, GridSearchCV)  
- Jupyter Notebook

---



