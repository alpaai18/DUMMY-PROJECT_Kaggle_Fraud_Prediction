# Libraries
import pandas as pd
import numpy as np
import time
import joblib

from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier

from src.preprocessing import cleaning_data
from src.feature_engineering import reduce_feature

# Test Data
path_data = r'Data\result\03_Exploratory_Data_Analysis_Test.csv'
df_test = pd.read_csv(path_data)

# Import Models
scaler = joblib.load('Output/Model/model_standard_scaler.pkl')
encoder = joblib.load('Output/Model/model_encoder_ordinal.pkl')

cb_model_v3 = CatBoostClassifier()
cb_model_v3.load_model("Output/Model/model_v3_catboost.cbm")

# Standardize amount columns
df_test['Amount'] = scaler.transform(df_test[['Amount']])

# feature_engineering
df_test = reduce_feature(df_test)

# Encode str columns
df_test[df_test.select_dtypes(include=['str']).columns.tolist()] = encoder.transform(df_test[df_test.select_dtypes(include=['str']).columns.tolist()])

# Cleaning Test Data
df_test = cleaning_data(df_test)

# Prepare Data for Model Testing
X_test = df_test.drop(columns='Class')
y_test = df_test['Class']

# Prediction Model
start_cb_test = time.perf_counter()                     # --> Time Started
cb_pred = cb_model_v3.predict_proba(X_test)[:, 1]       # --> Model do Pred
end_cb_test = time.perf_counter()                       # --> Time Stopped
test_cb_model_time = end_cb_test - start_cb_test        # --> Calculated pred time
cb_v3_auprc = average_precision_score(y_test, cb_pred)  # --> Get Metrics Model

# Export prediction to df_test
df_test['model_pred'] = cb_pred
df_test['model_class'] = (df_test['model_pred'] >= 0.5).astype(int)

df_test.to_csv(r'Output\Prediction\df_final.csv',index=False)

# Result
print()
print('='*73)
print('||' + f"Test (Inference) time CatBoost Classifier Model: {test_cb_model_time:.4f} seconds\t" +'||')
print('||' + (' '*70) + '||')
print('||' + 'Score AUPRC CatBoost Classifier Model \t:',  cb_v3_auprc, '\t' +'||')
print('='*73)
print()