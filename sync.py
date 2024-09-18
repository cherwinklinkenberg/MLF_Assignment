# %%
import tensorflow as tf
import pandas as pd
import numpy as np

from scipy.stats import uniform, randint

from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb


# %%
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from cleanlab.filter import find_label_issues

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X = train_df.drop('target', axis=1) 
y = train_df['target']              

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# %%

xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='error')
fit = xgb_model.fit(X_train, y_train)
importances = fit.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

top_n = 50
top_features = feature_importance_df['Feature'].head(top_n).values

# Create a new training set with only the top N features
X_train_top_features = X_train[top_features]
X_val_top_features = X_val[top_features]


# %%
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [ 0.1, 0.15, 0.2],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'reg_lambda': [0, 0.1, 1]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train_top_features, y_train)

# %%
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val_top_features)

val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy:.4f}')
print('Validation Classification Report:')
print(classification_report(y_val, y_val_pred))
print('Validation Confusion Matrix:')
print(confusion_matrix(y_val, y_val_pred))

X_test = test_df.drop('target', axis=1, errors='ignore')
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
X_test_top_feature = X_test[top_features] 

y_test_pred = best_model.predict(X_test_top_feature)
np.savetxt('predictions.txt', y_test_pred, fmt='%d', delimiter='')

print('Predictions have been saved to predictions.txt.')



