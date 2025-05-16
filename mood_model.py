import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

class MoodModel:
    def __init__(self):
        self.model = RandomForestRegressor()
        self.trained = False

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])

        # Extract date-based features
        df["Day_Of_Week"] = df["Date"].dt.weekday  # 0=Monday, 6=Sunday
        df["Is_Weekend"] = df["Day_Of_Week"].isin([5, 6]).astype(int)

        # Days since last entry
        df = df.sort_values("Date")
        df["Days_Since_Last_Log"] = df["Date"].diff().dt.days.fillna(1).clip(1, 7)

        return df

    def train(self, df: pd.DataFrame, test_size=0.2):
        df = self.add_features(df)

        features = [
            "Sleep_Hours", "Screen_Time_Hours", "Exercise",
            "Day_Of_Week", "Is_Weekend", "Days_Since_Last_Log"
        ]
        target = "Mood"

        X = df[features]
        y = df[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        self.model.fit(self.X_train, self.y_train)
        self.trained = True
        print("✅ Model trained with extended features")

    def evaluate(self):
        if not self.trained:
            raise RuntimeError("Train the model before evaluating.")

        y_pred = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(f"📊 MAE: {mae:.2f}")
        print(f"📈 R²: {r2:.2f}")
        return {"mae": mae, "r2": r2}

    def predict(self, sleep, screen_time, exercise, date=None, last_log_date=None):
        if not self.trained:
            raise RuntimeError("Train the model before predicting.")

        if date is None:
            date = pd.Timestamp.today()
        else:
            date = pd.to_datetime(date)

        if last_log_date is None:
            last_log_date = date - pd.Timedelta(days=1)
        else:
            last_log_date = pd.to_datetime(last_log_date)

        day_of_week = date.weekday()
        is_weekend = int(day_of_week in [5, 6])
        days_since_last_log = max((date - last_log_date).days, 1)

        input_df = pd.DataFrame([[
            sleep, screen_time, exercise,
            day_of_week, is_weekend, days_since_last_log
        ]], columns=[
            "Sleep_Hours", "Screen_Time_Hours", "Exercise",
            "Day_Of_Week", "Is_Weekend", "Days_Since_Last_Log"
        ])

        return self.model.predict(input_df)[0]

    def save_model(self, path="mood_model.pkl"):
        if not self.trained:
            raise RuntimeError("Train the model before saving.")
        joblib.dump(self.model, path)
        print(f"💾 Model saved to {path}")

    def load_model(self, path="mood_model.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        self.model = joblib.load(path)
        self.trained = True
        print(f"📂 Model loaded from {path}")
