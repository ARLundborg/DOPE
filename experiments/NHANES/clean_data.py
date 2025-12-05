import argparse

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer as Imputer

### First few lines clean data as in Lundberg et al. (2020)
### Columns with >50% missing are removed in _removed data (not in Lundberg)


df = pd.read_csv("experiments/NHANES/data/nhanes.txt")
y = df.pop("y").to_numpy()
X = df.copy()

# clean up a bit
for c in X.columns:
    if c.endswith("_isBlank"):
        del X[c]
X["bmi"] = (
    10000
    * X["weight"].values.copy()
    / (X["height"].values.copy() * X["height"].values.copy())
)
del X["weight"]
del X["height"]
del X["urine_hematest_isTrace"]  # would have no variance in the strain set
del X["SGOT_isBlankbutapplicable"]  # would have no variance in the strain set
del X[
    "calcium_isBlankbutapplicable"
]  # would have no variance in the strain set
del X[
    "uric_acid_isBlankbutapplicable"
]  # would only have one true value in the train set
del X[
    "urine_hematest_isVerylarge"
]  # would only have one true value in the train set
del X[
    "total_bilirubin_isBlankbutapplicable"
]  # would only have one true value in the train set
del X[
    "alkaline_phosphatase_isBlankbutapplicable"
]  # would only have one true value in the train set
del X["hemoglobin_isUnacceptable"]  # redundant with hematocrit_isUnacceptable
rows = np.where(
    np.invert(np.isnan(X["systolic_blood_pressure"]) | np.isnan(X["bmi"]))
)[0]
X = X.iloc[rows, :]
y = y[rows]


alot_of_missing = []
for col in X.columns:
    if np.isnan(X[col]).mean() > 0.5:
        alot_of_missing.append(col)
X_removed = X.drop(alot_of_missing, axis=1)


def impute_and_return(X, y):
    imp = Imputer()
    imp.fit(X)
    X_imp = imp.transform(X)

    X1 = pd.DataFrame(X_imp, columns=X.columns)
    df = pd.concat([X1, pd.Series(y > 0, dtype=np.int8, name="death")], axis=1)

    # Define target
    sbp = "systolic_blood_pressure"
    puls_pres = df.pop("pulse_pressure")
    df.insert(0, "high_pulse_pressure", ((puls_pres > 60)).astype(np.int16))

    # Standardize
    for col in df.columns:
        if len(df[col].unique()) > 2:
            df[col] = (df[col] - df[col].mean()) / df[col].var()
    return df


df = impute_and_return(X, y)
df.to_pickle("experiments/NHANES/data/NHANES_imputed.pkl")
df_removed = impute_and_return(X_removed, y)
df_removed.to_pickle("experiments/NHANES/data/NHANES_imputed.pkl")
