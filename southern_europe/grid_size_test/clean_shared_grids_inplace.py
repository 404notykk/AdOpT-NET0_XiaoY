#!/usr/bin/env python3
"""
clean_shared_grids_inplace.py

- Loads soil / anthro / morpho CSVs from the ../greece_data/geographical_feature folder
- Finds the intersection of GRID_ID across all three
- Filters each dataset to shared IDs only
- Overwrites the original CSV files in place (keeps .bak backups for safety)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# -------------------------
# Import data section
# -------------------------
path_data_case_study = Path("../italy_data")
path_files_grids = path_data_case_study / "geographical_feature"

soil_file   = path_files_grids / "soil_type_grids_italy_10km.csv"
anthro_file = path_files_grids / "anthropisation_grids_italy_10km.csv"
morpho_file = path_files_grids / "morphological_feature_grids_italy_10km.csv"

soil_data   = pd.read_csv(soil_file)
anthro_data = pd.read_csv(anthro_file)
morpho_data = pd.read_csv(morpho_file)

# -------------------------
# Config
# -------------------------
ID_COL = "GRID_OID"
MAKE_BACKUP = True   # set to False if you don't want .bak backups

# -------------------------
# Helpers
# -------------------------
def normalize_id(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan})
    return s

def prepare_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if ID_COL not in df.columns:
        raise KeyError(f"'{ID_COL}' not found in {name}. Columns: {list(df.columns)[:10]}")
    before = len(df)
    df[ID_COL] = normalize_id(df[ID_COL])
    df = df.dropna(subset=[ID_COL])
    dup_count = df.duplicated(subset=[ID_COL]).sum()
    df = df.drop_duplicates(subset=[ID_COL], keep="first")
    print(f"[{name}] rows {before} → {len(df)} (removed NaN IDs; {dup_count} duplicate IDs)")
    return df

def overwrite_csv(df: pd.DataFrame, file: Path):
    if MAKE_BACKUP and file.exists():
        bak = file.with_suffix(file.suffix + ".bak")
        os.replace(file, bak)
        print(f"Backup saved: {bak}")
    df.to_csv(file, index=False)
    print(f"[OK] Overwrote {file.name} with {len(df)} rows")

# -------------------------
# Cleaning
# -------------------------
print("Preparing datasets...")

soil_data   = prepare_df(soil_data,   soil_file.name)
anthro_data = prepare_df(anthro_data, anthro_file.name)
morpho_data = prepare_df(morpho_data, morpho_file.name)

# Find shared GRID_IDs
shared_ids = set(soil_data[ID_COL]) & set(anthro_data[ID_COL]) & set(morpho_data[ID_COL])
print(f"[Shared {ID_COL}] count: {len(shared_ids)}")

if len(shared_ids) == 0:
    raise RuntimeError("No shared IDs found! Are these CSVs from the same fishnet size?")

# Filter
soil_data_f   = soil_data[soil_data[ID_COL].isin(shared_ids)].copy()
anthro_data_f = anthro_data[anthro_data[ID_COL].isin(shared_ids)].copy()
morpho_data_f = morpho_data[morpho_data[ID_COL].isin(shared_ids)].copy()

# Overwrite originals
overwrite_csv(soil_data_f,   soil_file)
overwrite_csv(anthro_data_f, anthro_file)
overwrite_csv(morpho_data_f, morpho_file)

print("Done. All three CSVs now only contain shared GRID_IDs.")
