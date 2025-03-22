
# 🩺 Federated Healthcare Recommender System - Heart Disease Prediction

## ✅ **Project Overview**
This project builds a **personalized healthcare recommender system** focused on heart disease prediction.  
It integrates **Machine Learning models**, **data preprocessing**, **visualizations**, and **hyperparameter tuning**.  
✅ Future-ready for **Federated Learning simulation** to protect patient data privacy.

---

## 📂 **Dataset**
- **Source:** Kaggle - Heart Disease Dataset
- **Records:** 1025 patient records
- **Features:** 13 clinical attributes like age, blood pressure, cholesterol, chest pain type, etc.
- **Target:** 1 (Heart Disease Present) or 0 (No Heart Disease)

✅ Data cleaning, one-hot encoding, scaling, and correlation analysis completed.

---

## 🤖 **Machine Learning Models**
| Model                  | Accuracy   | Notes  |
|------------------------|-----------|--------|
| Logistic Regression    | 74.14%    | Baseline model |
| Random Forest (Tuned)  | 100%      | Hyperparameter tuned, potential overfitting |

- **Grid Search Tuning** applied to Random Forest
- **Cross-validation score for Logistic Regression:** 80.59%

✅ Models saved as `.pkl` files for deployment.

---

## 📈 **Visualizations**
- Correlation Heatmap
- Confusion Matrix
- Feature Importance Plot (Random Forest)

---

## 🌐 **Deployment (Coming)**
- ✅ Streamlit app for predictions
- ✅ API ready (optional FastAPI version)

---

## 🛠 **Folder Structure**
```
federated-healthcare-recommender/
│
├── data/                      # Raw & cleaned data
├── preprocessing/             # Data cleaning, ML model, tuning
├── models/                    # Saved ML models (.pkl)
├── visualizations/            # Plots and charts
├── streamlit_app/             # Interactive UI (Coming)
├── docs/                      # Documentation, reports
└── README.md                  # Project overview
```

---

## 🚀 **How to Run**
```bash
# Preprocessing & Model Training
python preprocessing/data_preprocessing.py
python preprocessing/ml_model.py
python preprocessing/tuned_model.py

# Visualizations
python visualizations/visualize.py

# Streamlit App (coming)
streamlit run streamlit_app/app.py
```

---

## 💪 **What’s Next**
✅ Streamlit app  
✅ Federated Learning Split Simulation  
✅ Deployment ready  
✅ Video demo + LinkedIn post

---

## 📚 **Skills Demonstrated**
- Machine Learning
- Data Preprocessing
- Hyperparameter Tuning
- Model Evaluation
- Deployment Readiness
- Federated Learning Concept

---

## 🔥 **Author**
👨‍💻 Niraj Patil   

