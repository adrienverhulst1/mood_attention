# This is a sample Python script.
from mood_dataset_generator import MoodDatasetGenerator
from mood_visualizer import MoodVisualizer
from mood_model import MoodModel
from mood_model_benchmark import MoodModelBenchmark
from UserInputCollector import UserInputCollector
import pandas as pd
import os

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

LOG_PATH = "mood_log.csv"
MIN_ENTRIES_FOR_TRAINING = 10

# Press the green button in the gutter to run the script.
def main():
    # Generate dummy data
    #generator = MoodDatasetGenerator(num_days=360)
    #df = generator.generate()
    #print(df.head())
    #generator.save_csv()

    collector = UserInputCollector(
        registry_path="feature_registry.json",
        schema_path="schema_version.json"
    )

    entry = collector.collect_input()
    collector.save_entry(entry, log_path=LOG_PATH)

    # Visualize
    #visualizer = MoodVisualizer(df)
    #visualizer.plot_mood_over_time(show=False, save_path="mood_trend.png")
    #visualizer.plot_correlation_heatmap(show=False, save_path="correlation_heatmap.png")

    # Benchmark models
    #benchmark = MoodModelBenchmark(df)
    #results = benchmark.run_benchmark()
    #benchmark.print_results()

    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        print(f"📝 Log updated. Total entries: {len(df)}")

        if len(df) < MIN_ENTRIES_FOR_TRAINING:
            print(f"⚠️ Not enough data for training (need at least {MIN_ENTRIES_FOR_TRAINING} entries).")
        else:
            print(f"✅ Ready for training when you are!")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
