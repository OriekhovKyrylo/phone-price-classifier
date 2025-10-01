# All imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score
from xgboost import XGBClassifier
import joblib
import os

#LOAD AND EDA

def eda(data):
    total_na = data.isna().sum().sum()
    print(f'Dimensions: {data.shape[0]} rows, {data.shape[1]} columns')
    print(f'Total number of NAs: {total_na}')
    print("%38s %10s    %10s %10s" % ("Column Name", "Data Type", "Count Distinct", "NA Values"))
    col_name = data.columns
    dtypes = data.dtypes
    uniq = data.nunique()
    na_val = data.isna().sum()
    for i in range(len(data.columns)):
        print("%38s %10s    %10s %10s" % (col_name[i], dtypes.iloc[i], uniq.iloc[i], na_val.iloc[i]))


df = pd.read_csv("data/my_phone.csv")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

eda(df)

result = df[df['price_range'].isnull()].copy()
df_train = df[df['price_range'].notnull()].copy()

print(f"Training data: {len(df_train)} rows")
print(f"Data to predict: {len(result)} rows")

df_train = df_train.drop("Unnamed: 0", axis=1)
if len(result) > 0:
    result = result.drop("Unnamed: 0", axis=1)

# DATA CLEANING

df_train['bluetooth'] = df_train['bluetooth'].replace({'NO': 0, 'YES': 1}).astype(int)
df_train['dual_sim'] = df_train['dual_sim'].replace({'NO': 0, 'YES': 1}).astype(int)
df_train['n_cores'] = df_train['n_cores'].round().astype(int)
df_train['price_range'] = df_train['price_range'].round().astype(int)

print("Price range distribution:")
pct_dist = (df_train['price_range'].value_counts(normalize=True).sort_index() * 100).round(1)
for price, pct in pct_dist.items():
    print(f"  Price {price}: {pct}%")

numerical_cols = ["battery_power", "weight", "memory", "ram", "pixel_height", "pixel_width"]

print("\nNumerical features statistics:")
print(df_train[numerical_cols].describe())

plt.figure(figsize=(12, 10))
correlation_matrix = df_train[numerical_cols + ['price_range']].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

print("\nCorrelation with price_range:")
target_corr = correlation_matrix['price_range'].sort_values(ascending=False)
print(target_corr)

df_train['aspect_ratio'] = df_train['pixel_height'] / (df_train['pixel_width'] + 1)
df_train['is_flagship'] = ((df_train['ram'] > 6000) & (df_train['battery_power'] > 3000)).astype(int)

# TRAIN-VALIDATION-TEST SPLIT

X = df_train.drop('price_range', axis=1)
y = df_train['price_range']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\nData split:")
print(f"  Train: {X_train.shape[0]} samples")
print(f"  Validation: {X_val.shape[0]} samples")
print(f"  Test: {X_test.shape[0]} samples")


def logreg_pipeline(X_train, y_train):
    logreg_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    logreg_model.fit(X_train, y_train)
    return logreg_model


def randomforest_pipeline(X_train, y_train):
    rf_model = make_pipeline(RandomForestClassifier(n_estimators=100, random_state=42))
    rf_model.fit(X_train, y_train)
    return rf_model


def xgboost_pipeline(X_train, y_train):
    xgboost_model = make_pipeline(XGBClassifier(random_state=42, n_estimators=100))
    xgboost_model.fit(X_train, y_train)
    return xgboost_model


pipelines = {
    'logreg': logreg_pipeline,
    'randomforest': randomforest_pipeline,
    'xgboost': xgboost_pipeline,
}

models = {}
print("\nTraining models...")
for model_name, model_pipeline in pipelines.items():
    print(f"  {model_name} training...")
    models[model_name] = model_pipeline(X_train, y_train)

#MODEL EVALUATION

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix:\n{cm}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    try:
        y_pred_proba = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        print(f"ROC AUC score (OvR): {roc_auc:.4f}")
    except:
        print("ROC AUC not available for this model")



model_scores = {}
print("Evaluating models on test set:")
for model_name, model in models.items():
    print(f"{model_name} evaluation:")
    evaluate_model(model, X_test, y_test)
    model_scores[model_name] = accuracy_score(y_test, model.predict(X_test))

print("Model comparison:")
for model_name, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {model_name}: {score:.4f}")

best_model_name = max(model_scores, key=model_scores.get)
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name} with accuracy {model_scores[best_model_name]:.4f}")

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, best_model.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'{best_model_name.upper()} Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

#  MODEL SAVING
print("Visualizations saved: correlation_matrix.png, confusion_matrix.png")

if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(best_model, f'models/phone_price_{best_model_name}.pkl')
print(f"Model saved: models/phone_price_{best_model_name}.pkl")

if len(result) > 0:
    print(f"\nPredicting {len(result)} phones with missing price ranges...")

    result['bluetooth'] = result['bluetooth'].map({'NO': 0, 'YES': 1})
    result['dual_sim'] = result['dual_sim'].map({'NO': 0, 'YES': 1})
    result['n_cores'] = result['n_cores'].round().astype(int)
    result['aspect_ratio'] = result['pixel_height'] / (result['pixel_width'] + 1)
    result['is_flagship'] = ((result['ram'] > 6000) & (result['battery_power'] > 3000)).astype(int)

    X_result = result.drop('price_range', axis=1)
    result['predicted_price_range'] = best_model.predict(X_result)

    result.to_csv('data/predicted_phone_prices.csv', index=False)
    print(f"Predictions saved to data/predicted_phone_prices.csv")

    print("\nSample predictions:")
    print(result[['battery_power', 'ram', 'memory', 'predicted_price_range']].head(10))

print("PIPELINE COMPLETED SUCCESSFULLY")
