import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config_breast import DATA_PATH, RANDOM_STATE


def load_and_preprocess():
    # 1. Load dataset
    df = pd.read_csv(DATA_PATH)

    # 2. Drop ID column
    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)

    # 3. Encode target 
    if "diagnosis" not in df.columns:
        raise ValueError("Expected 'diagnosis' column not found.")

    df["diagnosis"] = (df["diagnosis"] == "M").astype(int)

    # 4. Fill missing values
    df = df.fillna(df.median(numeric_only=True))

    # 5. Feature selection 
    corr = df.corr()
    cor_target = abs(corr["diagnosis"])
    relevant_features = cor_target[cor_target > 0.2]
    feature_names = [
        col for col in relevant_features.index if col != "diagnosis"
    ]

    # 6. Prepare X, y
    X = df[feature_names].values
    y = df["diagnosis"].values

    # 7. Stable train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # 8. Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 9. Return
    return X_train, X_test, y_train, y_test, feature_names
