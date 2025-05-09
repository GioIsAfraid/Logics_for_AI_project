import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path, sep=';'):
    """
    Load dataset from CSV.
    """
    df = pd.read_csv(path, sep=sep)
    return df

def preprocess_data(df):
    """
    Preprocess dataset:
    - Binarize target
    - Drop unused columns
    - One-hot encoding
    - Train-test split
    """

    if 'pass' not in df.columns:
        df['pass'] = (df['G3'] >= 10).astype(int)
        df = df.drop(columns=['G1', 'G2', 'G3'])

    original = df.copy()

    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop(columns='pass')
    y = df_encoded['pass']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42
    )
    original_test = original.iloc[X_test.index]

    return X_train, X_test, y_train, y_test, original_test, original
