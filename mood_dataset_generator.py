import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

class MoodDatasetGenerator:
    def __init__(self, num_days=60, seed=42):
        self.num_days = num_days
        self.seed = seed
        self.df = None
        self._set_random_seed()

    def _set_random_seed(self):
        np.random.seed(self.seed)
        random.seed(self.seed)

    def _generate_dates(self):
        start_date = datetime.today() - timedelta(days=self.num_days)
        all_dates = [start_date + timedelta(days=i) for i in range(self.num_days)]

        # Simulate skipped entries (e.g., 10–20% of days missing)
        skip_fraction = 0.15  # adjust as needed
        num_to_skip = int(self.num_days * skip_fraction)
        skip_indices = sorted(random.sample(range(self.num_days), num_to_skip))

        dates_with_skips = [d for i, d in enumerate(all_dates) if i not in skip_indices]
        return dates_with_skips

    def generate_sleep(self, screen_time_array):
        # Screen time 2–12 hours → Sleep hours 8–4 hours (inverse)
        scaled_screen = (12 - screen_time_array) / 2  # 2h → 5, 12h → 0
        noise = np.random.normal(0, 1.0, len(screen_time_array))
        sleep_hours = scaled_screen + noise + 3  # shift so typical value centers near 7
        return np.clip(sleep_hours, 0, 8)

    def generate_mood(self, sleep_hours_array, exercise_array):
        scaled_sleep = (sleep_hours_array - 4) / 2  # 0–8h → [-2, +2]
        exercise_bonus = 0.5 * np.array(exercise_array)
        noise = np.random.normal(0, 0.8, len(sleep_hours_array))
        mood = scaled_sleep + exercise_bonus + noise
        return np.clip(mood, -3, 3).round()

    def _generate_data(self):
        dates = self._generate_dates()
        n = len(dates)

        exercise = [random.choice([1, 0]) for _ in range(n)]
        screen_time = np.clip(np.random.normal(8, 2, n), 2, 16)

        sleep_hours = self.generate_sleep(screen_time)
        mood = self.generate_mood(sleep_hours, exercise)

        return pd.DataFrame({
            "Date": dates,
            "Sleep_Hours": sleep_hours,
            "Exercise": exercise,
            "Screen_Time_Hours": screen_time,
            "Mood": mood
        })

    def generate(self):
        self.df = self._generate_data()
        return self.df

    def save_csv(self, filename="mood_tracking_dataset.csv"):
        if self.df is None:
            self.generate()
        self.df.to_csv(filename, index=False)
        print(f"✅ Dataset saved to '{filename}'")
