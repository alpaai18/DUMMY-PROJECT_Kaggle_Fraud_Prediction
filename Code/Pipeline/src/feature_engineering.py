import numpy as np

def feature_engineering(df):
    df['Bucket_Amount'] = np.where(df['Amount'] > 0.683, '> 0.683', '<= 0.683')
    df['Bucket_V14']    = np.where(df['V14'] < -17.23, '< -17.23', '>= -17.23')
    df['Bucket_V9']     = np.where(df['V9'] < -8.09, '< -8.09', '>= -8.09')
    df['Bucket_V7']     = np.where(df['V7'] < -17.68, '< -17.68', '>= -17.68')
    df['Bucket_V12']    = np.where(df['V12'] < -17.23, '< -17.23', '>= -17.23')
    df['Bucket_V1']     = np.where(df['V1'] < -13.68, '< -13.68', '>= -13.68')
    df['Bucket_V10']    = np.where(df['V10'] < -14.92, '< -14.92', '>= -14.92')
    df['Bucket_V24']    = np.where(df['V24'] < -1.511, '< -1.511', '>= -1.511')
    df['Bucket_V3']     = np.where(df['V3'] < -17.71, '< -17.71', '>= -17.71')

    return df

def reduce_feature(df):
    feature_engineering(df)
    df = df[[
        'Bucket_Amount', 'Bucket_V14', 'Bucket_V9', 'Bucket_V7', 'Bucket_V12', 
        'Bucket_V1', 'Bucket_V10', 'Bucket_V24', 'Bucket_V3', 'V14', 'V17', 'V12', 
        'V11', 'V16', 'V10', 'V4', 'V3', 'V18', 'V9', 'V26', 'V1', 'V8', 'Amount', 
        'V15', 'Time', 'V20', 'Class'
    ]]

    return df