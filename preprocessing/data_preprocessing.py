import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Direct file path to avoid path confusion
DATA_FILE = '/Users/nirajpatil/Documents/Quant-Projects/federated-healthcare-recommender/federated-healthcare-recommender/data/heart.csv'
OUTPUT_FILE = '/Users/nirajpatil/Documents/Quant-Projects/federated-healthcare-recommender/federated-healthcare-recommender/data/cleaned_scaled_heart.csv'

print("✅ Final Path Used -->", DATA_FILE)
print("File exists?", os.path.exists(DATA_FILE))


def load_data():
    if os.path.exists(DATA_FILE):
        print("✅ Dataset loaded successfully.")
        return pd.read_csv(DATA_FILE)
    else:
        print("❌ Dataset not found at:", DATA_FILE)
        return None

def clean_and_scale(df):
    # Dropping any null values if present
    df.dropna(inplace=True)
    print(f"✅ Data after dropping NA: {df.shape}")

    # Optional one-hot encoding for 'cp' (chest pain)
    if 'cp' in df.columns:
        df = pd.get_dummies(df, columns=['cp'], drop_first=True)

    # Correlation Heatmap (numeric columns only)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('/Users/nirajpatil/Documents/Quant-Projects/federated-healthcare-recommender/federated-healthcare-recommender/data/correlation_heatmap.png')
    print("✅ Correlation heatmap saved.")

    # Separate target column
    target = df['target']

    # ✅ Select only numeric features for scaling
    numeric_features = df.drop('target', axis=1).select_dtypes(include='number')

    # Scaling numeric features only
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_features)

    # Recombine scaled features and target into final DataFrame
    df_scaled = pd.DataFrame(scaled, columns=numeric_features.columns)
    df_scaled['target'] = target.values

    print(f"✅ Final cleaned & scaled data shape: {df_scaled.shape}")
    return df_scaled


if __name__ == "__main__":
    df = load_data()
    if df is not None:
        cleaned_df = clean_and_scale(df)
        cleaned_df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Cleaned and scaled dataset saved to: {OUTPUT_FILE}")
