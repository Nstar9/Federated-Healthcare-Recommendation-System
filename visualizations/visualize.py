import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load model and data
rf_model = joblib.load("/Users/nirajpatil/Documents/Quant-Projects/federated-healthcare-recommender/federated-healthcare-recommender/models/tuned_random_forest.pkl")
df = pd.read_csv("/Users/nirajpatil/Documents/Quant-Projects/federated-healthcare-recommender/federated-healthcare-recommender/data/cleaned_scaled_heart.csv")

X = df.drop('target', axis=1)
y = df['target']
y_pred = rf_model.predict(X)

# Set custom aesthetic style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.facecolor'] = '#f5f5f5'

# ✅ Beautiful Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr', cbar=False,
                linewidths=1.5, linecolor='black',
                annot_kws={"size": 16, "weight": "bold"})
    plt.xticks([0.5, 1.5], ['No Disease', 'Disease'], fontsize=14)
    plt.yticks([0.5, 1.5], ['No Disease', 'Disease'], fontsize=14, rotation=0)
    plt.title('Confusion Matrix', fontsize=18, weight='bold', color='#333333')
    plt.tight_layout()
    plt.savefig("visualizations/confusion_matrix.png")
    print("✅ Styled Confusion Matrix saved!")

# ✅ Aesthetic Feature Importance Plot
def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    sorted_idx = importance.argsort()
    
    plt.figure(figsize=(10, 7))
    sns.barplot(x=importance[sorted_idx], y=[feature_names[i] for i in sorted_idx],
                palette="viridis", edgecolor=".6")
    
    plt.title('Feature Importance - Random Forest', fontsize=18, weight='bold', color='#333333')
    plt.xlabel('Importance Score', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.tight_layout()
    plt.savefig("visualizations/feature_importance.png")
    print("✅ Styled Feature Importance saved!")

# Run both
plot_confusion_matrix(y, y_pred)
plot_feature_importance(rf_model, X.columns)
