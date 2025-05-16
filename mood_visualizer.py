import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

class MoodVisualizer:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        self._prepare_data()

    def _prepare_data(self):
        if "Date" in self.df.columns:
            self.df["Date"] = pd.to_datetime(self.df["Date"])

    def plot_mood_over_time(self, show=True, save_path=None):
        plt.figure(figsize=(10, 4))
        sns.lineplot(data=self.df, x="Date", y="Mood", marker="o")
        plt.title("Mood Over Time")
        plt.xlabel("Date")
        plt.ylabel("Mood")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"✅ Mood trend saved to {save_path}")
        if show:
            plt.show()
        else:
            plt.close()

    def plot_correlation_heatmap(self, show=True, save_path=None):
        numeric_cols = ["Sleep_Hours", "Exercise", "Screen_Time_Hours", "Mood"]
        corr = self.df[numeric_cols].corr()

        plt.figure(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Between Features")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"✅ Correlation heatmap saved to {save_path}")
        if show:
            plt.show()
        else:
            plt.close()
