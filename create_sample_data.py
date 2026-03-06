"""
Create minimal sample CSV files so R01_supervised_random_forest.py can run
when no data is present. Replace with your real data (export from Google
Sheets / MASTER_SPREADSHEET) for actual analysis.
"""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Minimal columns matching what the pipeline expects (see data/README.md)
filtered_cols = [
    "Rat ID",
    "Intended Asphyxia/Asystole time (min)",
    "4hr NDS",
    "24hr NDS",
    "24morning NDS",
]
filtered_data = [
    ["RAT001", 7, "32(40)", "45(50)", "48"],
    ["RAT002", 7.5, "27", "60", "60"],
    ["RAT003", 8, "40(42)", "55", "58"],
    ["RAT004", 8, "25", "35", "38"],
    ["RAT005", 7, "38", "52", "52"],
]
pd.DataFrame(filtered_data, columns=filtered_cols).to_csv(
    os.path.join(DATA_DIR, "Filtered_Subset_No_REBOA.csv"), index=False
)

selected_cols = [
    "Rat ID",
    "Time from ROSC that BSR Reaches 0.5",
    "DBP_baseline",
    "SBP_baseline",
    "MAP_baseline",
]
selected_data = [
    ["RAT001", 10, 80, 120, 90],
    ["RAT002", 12, 82, 118, 92],
    ["RAT003", 11, 78, 122, 88],
    ["RAT004", 9, 85, 115, 95],
    ["RAT005", 10, 79, 121, 89],
]
pd.DataFrame(selected_data, columns=selected_cols).to_csv(
    os.path.join(DATA_DIR, "No_REBOA_selected_column.csv"), index=False
)

pd.DataFrame([["RAT001", ""], ["RAT002", ""], ["RAT003", ""], ["RAT004", ""], ["RAT005", ""]], columns=["Rat ID", "notes"]).to_csv(
    os.path.join(DATA_DIR, "cleaned_no_reboa.csv"), index=False
)

print("Created sample CSVs in data/:")
print("  Filtered_Subset_No_REBOA.csv, No_REBOA_selected_column.csv, cleaned_no_reboa.csv")
print("Run: python R01_supervised_random_forest.py")
print("Replace these with your real data when ready.")
