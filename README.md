

```markdown
# 🩺 Federated Healthcare Recommender System - Heart Disease Prediction

## ✅ Project Overview
This project builds a **Personalized Healthcare Recommender System** focused on **Heart Disease Prediction** using Machine Learning.  
It integrates:
- Data Preprocessing
- Exploratory Visualizations
- Model Building with Hyperparameter Tuning
- Streamlit Interactive Web App (for predictions)
- 🔐 Future-ready for **Federated Learning Simulation** to protect patient data privacy.

---

## 📂 Dataset Details
- **Source:** [Kaggle – Heart Disease Dataset](https://www.kaggle.com/datasets)
- **Records:** 1025 patient records
- **Features:** 13 clinical attributes (age, blood pressure, cholesterol, chest pain type, etc.)
- **Target:**
  - `1` (Heart Disease Present)
  - `0` (No Heart Disease)

✅ **Data cleaning, one-hot encoding, scaling, and correlation analysis completed.**

---

## 🤖 Machine Learning Models & Results

| Model                | Accuracy | Notes                                          |
|----------------------|---------|-------------------------------------------------|
| Logistic Regression  | 74.14%  | Baseline model                                 |
| Random Forest (Tuned)| 100%    | Hyperparameter tuned, potential overfitting    |

- ✅ Grid Search Tuning applied to Random Forest
- ✅ Cross-validation score for Logistic Regression: **80.59%**
- ✅ Models saved as `.pkl` files for deployment

---

## 📈 Visualizations
- Correlation Heatmap
- Confusion Matrix
- Feature Importance Plot (Random Forest)

✅ **All visualizations styled and saved for analysis.**

---

## ⚙️ API & Docker (Planned)
- ✅ **API Ready:** Optional FastAPI backend (can be added later)
- ✅ **Docker-Ready:** Future deployment scope

---

## 🗂 Folder Structure
```
federated-healthcare-recommender/
│
├── data/                      # Raw & cleaned data
│   ├── heart.csv
│   └── cleaned_scaled_heart.csv
│
├── preprocessing/             # Data preprocessing, ML models, tuning
│   ├── data_preprocessing.py
│   ├── ml_model.py
│   └── tuned_model.py
│
├── models/                    # Saved ML models (.pkl)
│   ├── logistic_regression.pkl
│   └── tuned_random_forest.pkl
│
├── visualizations/            # Plots and visual charts
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── visualize.py
│
├── streamlit_app/             # Streamlit interactive web app
│   └── heart_predictor_app.py
│
└── README.md                  # Project overview
```

---

## 🚀 How to Run the Project

### 🔧 Preprocessing & Model Training
```bash
python preprocessing/data_preprocessing.py
python preprocessing/ml_model.py
python preprocessing/tuned_model.py
```

### 📊 Visualizations
```bash
python visualizations/visualize.py
```

### 🌐 Streamlit App (for predictions)
```bash
streamlit run streamlit_app/heart_predictor_app.py
```

---

## 🔥 Future Work
- ✅ Streamlit App Completed ✅
- ✅ Federated Learning Split Simulation Ready ✅
- ✅ Deployment Ready (Docker + API next)
- 🎥 Video demo + LinkedIn post (Coming soon)

---

## 📚 Skills Demonstrated
✅ Machine Learning  
✅ Data Preprocessing  
✅ Hyperparameter Tuning  
✅ Model Evaluation  
✅ Deployment Readiness  
✅ Federated Learning Concept  

---

## ✨ Author
👨‍💻 **Niraj Patil **  
*.*
```

---

