# Alzheimer's Disease Detection — Experiment 2 (Preprocessing)

## Objective
Preprocess raw Alzheimer's disease dataset to obtain a cleaned version suitable for ML models.

## Steps
1. Load raw CSV (`data/raw/alzheimers_disease_data.csv`)
2. Handle missing values
3. Encode categorical variables
4. Scale numeric columns
5. Save cleaned dataset and preprocessing artifacts

## Outputs
- `data/processed/alzheimers_cleaned.csv`
- `artifacts/scaler.pkl`
- `artifacts/encoder.pkl`

Run preprocessing:
```bash
python src/preprocess.py
