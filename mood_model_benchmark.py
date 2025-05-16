import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

class MoodModelBenchmark:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results = []

    def _add_features(self, df):
        df["Date"] = pd.to_datetime(df["Date"])
        df["Day_Index"] = (df["Date"] - df["Date"].min()).dt.days
        df["Day_Of_Week"] = df["Date"].dt.weekday
        df["Is_Weekend"] = df["Day_Of_Week"].isin([5, 6]).astype(int)
        df = df.sort_values("Date")
        df["Days_Since_Last_Log"] = df["Date"].diff().dt.days.fillna(1).clip(1, 7)
        return df

    def run_benchmark(self):
        df = self._add_features(self.df)

        X = df[[
            "Sleep_Hours", "Screen_Time_Hours", "Exercise",
            "Day_Of_Week", "Is_Weekend", "Days_Since_Last_Log"
        ]]
        y = df["Mood"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Models and hyperparameters to tune
        model_params = {
            "LinearRegression": {
                "model": LinearRegression(),
                "params": {}  # no tuning needed
            },
            "RandomForest": {
                "model": RandomForestRegressor(),
                "params": {
                    "n_estimators": [50, 100],
                    "max_depth": [3, 5, None]
                }
            },
            "KNN": {
                "model": KNeighborsRegressor(),
                "params": {
                    "n_neighbors": [3, 5, 7, 9]
                }
            },
            "GradientBoosting": {
                "model": GradientBoostingRegressor(),
                "params": {
                    "n_estimators": [25, 50, 100],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5, 7]
                }
            }
        }

        self.results = []

        for name, config in model_params.items():
            print(f"🎲 Random search for {name}...")

            search = RandomizedSearchCV(
                config["model"],
                config["params"],
                n_iter=10,  # Try 10 random combinations
                cv=3,
                scoring="neg_mean_absolute_error",
                random_state=42,
                n_jobs=-1
            )
            search.fit(X_train, y_train)

            best_model = search.best_estimator_
            y_pred = best_model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            self.results.append({
                "Model": name,
                "Best Params": search.best_params_,
                "MAE": round(mae, 3),
                "R²": round(r2, 3)
            })

        return pd.DataFrame(self.results).sort_values("MAE")

    def print_results(self):
        df_results = pd.DataFrame(self.results)
        print("\n📊 Benchmark Results:")
        print(df_results.sort_values("MAE").to_string(index=False))
