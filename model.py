import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def train_model(df):
    try:
        df = df.copy()

        # Clean data
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

        df = df.dropna()

        # Minimum data check
        if len(df) < 5:
            return None

        # Feature engineering
        df['day'] = df['date'].dt.day

        X = df[['day']]
        y = df['amount']

        model = LinearRegression()
        model.fit(X, y)

        return model

    except Exception:
        return None


def predict_future(model, days=7):
    try:
        if model is None:
            return None

        future_days = np.arange(1, days + 1).reshape(-1, 1)
        predictions = model.predict(future_days)

        return predictions

    except Exception:
        return None