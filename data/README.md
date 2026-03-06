# Data for R01_supervised_random_forest.py

**Option A – CSV files** (export from Google Sheets or your source):

- **Filtered_Subset_No_REBOA.csv** – subset with NDS columns: `4hr NDS`, `24hr NDS`, `24morning NDS`, `Rat ID`, `Intended Asphyxia/Asystole time (min)`
- **No_REBOA_selected_column.csv** – selected columns for features (same Rat IDs as above), including DBP/SBP/MAP columns and `Rat ID`, `Time from ROSC that BSR Reaches 0.5`
- **cleaned_no_reboa.csv** – (optional) cleaned data; only the first two are required for the pipeline

**Option B – Single Excel (.xlsx) file**

- Put one `.xlsx` file in this folder with **three sheets** named exactly: `Filtered_Subset_No_REBOA`, `No_REBOA_selected_column`, `cleaned_no_reboa` (same as the notebook’s Google Sheet structure). The script will use it if no CSV files are found. Requires `openpyxl`: `pip install openpyxl`

Run from repo root with:

```bash
conda activate eightsleep-ml
python R01_supervised_random_forest.py
```

Outputs are written to `outputs/`.
