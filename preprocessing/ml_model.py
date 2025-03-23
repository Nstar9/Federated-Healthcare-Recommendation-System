import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the cleaned dataset
DATA_FILE = '/Users/nirajpatil/Documents/Quant-Projects/federated-healthcare-recommender/federated-healthcare-recommender/data/cleaned_scaled_heart.csv'
df = pd.read_csv(DATA_FILE)
print("✅ Cleaned data loaded:", df.shape)

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data split: Train ->", X_train.shape, "Test ->", X_test.shape)

# Logistic Regression Model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Random Forest Classifier Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluation Function
def evaluate_model(name, y_test, y_pred):
    print(f"\n🔎 {name} Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Evaluate Both Models
evaluate_model("Logistic Regression", y_test, y_pred_log)
evaluate_model("Random Forest Classifier", y_test, y_pred_rf)
