# AqSolDB-Solubility-RF  
Predicting aqueous solubility (logS) with Random Forest + MACCS fingerprints.

## Dataset
- [AqSolDB](https://github.com/mcsorkun/AqSolDB) – 9,982 unique compounds  
- Target: logS (μmol/L)

## Model
- Features: 167-bit MACCS keys (RDKit)  
- Algorithm: Random Forest Regressor (n_estimators=500)  
- Train/Test Split: 80/20 random split  
- Metric: RMSE = 0.68

## Quick Start
```bash
git clone https://github.com/YOUR_USER_NAME/AqSolDB-Solubility-RF.git
cd AqSolDB-Solubility-RF
conda env create -f environment.yml  # optional
python solubility_rf.py
