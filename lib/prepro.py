import pandas as pd

# データのクレンジングと前処理
def clean_data(df):
    df.rename(columns={'gender': 'Gender', 'tenure': 'Tenure', 'customerID': 'CustomerID'}, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    df['Partner'] = df['Partner'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['Dependents'] = df['Dependents'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['PhoneService'] = df['PhoneService'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['MultipleLines'] = df['MultipleLines'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['OnlineSecurity'] = df['OnlineSecurity'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['OnlineBackup'] = df['OnlineBackup'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['DeviceProtection'] = df['DeviceProtection'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['TechSupport'] = df['TechSupport'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['StreamingTV'] = df['StreamingTV'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['StreamingMovies'] = df['StreamingMovies'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['PaperlessBilling'] = df['PaperlessBilling'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['InternetServiceType'] = df['InternetService'].apply(lambda x: {'No': 0, 'DSL': 1, 'Fiber optic': 2}[x])
    df['InternetService'] = df['InternetService'].apply(lambda x: 1 if x != 'No' else 0)
    df['IsContracted'] = df['Contract'].apply(lambda x: 0 if x == 'Month-to-month' else 1)
    df['ContractType'] = df['Contract'].apply(lambda x: {'Month-to-month': 0, 'One year': 1, 'Two year': 2}[x])
    df = df.drop(columns=['Contract'])
    df['PaymentMethod'] = df['PaymentMethod'].apply(lambda x: {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}[x])
    return df

def one_hot_encoding(df, columns):
    return pd.get_dummies(df, columns=columns, dummy_na=False)

def merge_df(*dfs):
    return pd.concat(dfs, axis=1)