import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):

    df = df.copy()

    label_cols = [
        'gender',
        'married',
        'education',
        'self_employed',
        'property_area'
    ]

    le = LabelEncoder()

    for col in label_cols:
        df[col] = le.fit_transform(df[col])

    df['dependents'] = df['dependents'].replace('3+',3)
    df['dependents'] = df['dependents'].astype(int)

    df['loan_status'] = df['loan_status'].map({'Y':1,'N':0})

    df["property_area"] = df["property_area"].replace({
        "Rural":0,
        "Semiurban":1,
        "Urban":2,
        "None":3
    })

    return df