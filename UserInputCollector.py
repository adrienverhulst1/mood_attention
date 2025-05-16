import json
from datetime import datetime
import os
import pandas as pd

class UserInputCollector:
    def __init__(self, registry_path="feature_registry.json", schema_path="schema_version.json"):
        self.registry_path = registry_path
        self.schema_version = self.load_schema_version(schema_path)
        self.registry = self.load_registry(registry_path)
        self.active_features = self.get_active_features()

    def load_schema_version(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("version", "unknown")

    def load_registry(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def get_active_features(self, current_date=None):
        current_date = current_date or datetime.today()
        active = []
        for f in self.registry:
            intro = datetime.fromisoformat(f["introduced_on"])
            retired = datetime.fromisoformat(f["retired_on"]) if f.get("retired_on") else None
            if intro <= current_date and (retired is None or current_date <= retired):
                active.append(f)
        return active

    def save_entry(self, entry, log_path="mood_log.csv"):
        # Load existing log or create new one
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
        else:
            df = pd.DataFrame()

        # Align entry to existing columns
        entry_df = pd.DataFrame([entry])

        # Combine, align all columns
        combined = pd.concat([df, entry_df], ignore_index=True).fillna("")
        combined.to_csv(log_path, index=False)
        print(f"✅ Entry saved to {log_path}")

    def collect_input(self):
        entry = {"date": datetime.today().date().isoformat()}
        entry["schema_version"] = self.schema_version
        for feature in self.active_features:
            fname = feature["name"]
            ftype = feature["type"]

            if ftype == "float" or ftype == "int":
                min_val = feature.get("min", 0)
                max_val = feature.get("max", 10)
                val = float(input(f"{fname} ({min_val}-{max_val}): "))
                entry[fname] = int(val) if ftype == "int" else val

            elif ftype == "bool":
                val = input(f"{fname}? (yes/no): ").lower() in ["yes", "y", "1"]
                entry[fname] = int(val)

            elif ftype == "str":
                val = input(f"{fname}: ")
                entry[fname] = val

        return entry
