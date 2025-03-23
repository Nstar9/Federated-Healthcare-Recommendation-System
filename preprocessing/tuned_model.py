import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load Cleaned Data
DATA_FILE = '/Users/nirajpatil/Documents/Quant-Projects/federated-healthcare-recommender/federated-healthcare-recommender/data/cleaned_scaled_heart.csv'
df = pd.read_csv(DATA_FILE)
print("✅ Cleaned data loaded:", df.shape)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Data split done.")

# ✅ Random Forest Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"✅ Best Random Forest Params: {grid_search.best_params_}")
best_rf = grid_search.best_estimator_

# ✅ Evaluate Tuned Random Forest
y_pred_rf = best_rf.predict(X_test)
print("\n🔎 Tuned Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ✅ Optional: Cross-validation for Logistic Regression
log_model = LogisticRegression()
cv_scores = cross_val_score(log_model, X, y, cv=5, scoring='accuracy')
print(f"✅ Logistic Regression Cross-Validation Accuracy: {cv_scores.mean():.4f}")

# ✅ Save Models
joblib.dump(best_rf, '/Users/nirajpatil/Documents/Quant-Projects/federated-healthcare-recommender/federated-healthcare-recommender/models/tuned_random_forest.pkl')
joblib.dump(log_model.fit(X_train, y_train), '/Users/nirajpatil/Documents/Quant-Projects/federated-healthcare-recommender/federated-healthcare-recommender/models/logistic_regression.pkl')
print("✅ Models saved successfully.")
