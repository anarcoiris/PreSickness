"""Quick test script."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

df = pd.read_parquet('data/processed/paciente1/training_dataset_clusters.parquet')
print(f'Dataset: {df.shape}')

exclude = ['date', 'first_message', 'last_message', 'relapse_in_7d', 'relapse_in_14d', 'relapse_in_30d']
features = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]
print(f'Features: {len(features)}')

X = df[features].fillna(0)
y = df['relapse_in_14d']
print(f'Positive rate: {y.mean():.2%}')

split = int(len(X) * 0.8)
X_train, X_val = X.iloc[:split], X.iloc[split:]
y_train, y_val = y.iloc[:split], y.iloc[split:]
print(f'Train: {len(X_train)}, Val: {len(X_val)}')

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=10, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
proba = model.predict_proba(X_val)[:, 1]

auroc = roc_auc_score(y_val, proba)
auprc = average_precision_score(y_val, proba)
print(f'\nRandomForest: AUROC={auroc:.4f}, AUPRC={auprc:.4f}')

# Feature importance
import numpy as np
importances = list(zip(features, model.feature_importances_))
importances.sort(key=lambda x: x[1], reverse=True)
print('\nTop 5 features:')
for f, i in importances[:5]:
    print(f'  {f}: {i:.4f}')
