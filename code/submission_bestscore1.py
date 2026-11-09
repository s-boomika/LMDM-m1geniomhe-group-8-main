import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# ===================================================
# CONFIG
# ===================================================
ID_COL = "trustii_id"
AGE_COL = "Age_Baseline"
SEX_COL = "Genre"
TARGET_SEVERITY = "OUTCOME SEVERITY"
TARGET_MACE = "OUTCOME MACE"

SNP_COLS_ALL = [f"SNP{i}" for i in range(1, 289)]
PRIORITY_SNP = [f"SNP{i}" for i in range(1, 76)]

# ===================================================
# LOAD DATA
# ===================================================
print("Loading data...")
train = pd.read_csv("/home/snouzlat/Téléchargements/learning_MDM/project/data/cardihack_final-train.csv")
test  = pd.read_csv("/home/snouzlat/Téléchargements/learning_MDM/project/data/cardihack_final-test.csv")
dummy = pd.read_csv("/home/snouzlat/Téléchargements/learning_MDM/project/data/cardihack_dummy_pred.csv")

# ===================================================
# SNP FILTERING (SAFE)
# ===================================================
def filter_by_maf(df, snp_cols, maf_threshold=0.01):
    selected = []
    for snp in snp_cols:
        vals = df[snp].dropna()
        if len(vals) < 10:
            continue
        freq = vals.value_counts()
        if len(freq) < 2:
            continue
        maf = freq.min() / freq.sum()
        if maf >= maf_threshold:
            selected.append(snp)
    return selected

def chi2_select_snps(df, target, snp_cols, p_threshold=0.1):
    selected = []
    for snp in snp_cols:
        tab = pd.crosstab(df[snp], df[target])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        try:
            _, p, _, _ = chi2_contingency(tab)
            if p < p_threshold:
                selected.append(snp)
        except:
            continue
    return selected

def enforce_priority(train, selected, max_total=90):
    if not any(snp in PRIORITY_SNP for snp in selected):
        backup = chi2_select_snps(train, TARGET_MACE, PRIORITY_SNP, 0.2)
        if backup:
            selected = [backup[0]] + selected
    return selected[:max_total]

# ===================================================
# FEATURE SELECTION
# ===================================================

# -------- SEVERITY: Age + Sex + MANY SNPs (polygenic)
sev_snps = filter_by_maf(train, SNP_COLS_ALL, maf_threshold=0.01)
sev_snps = chi2_select_snps(train, TARGET_SEVERITY, sev_snps, p_threshold=0.1)
sev_features = [AGE_COL, SEX_COL] + sev_snps

# -------- MACE: Clinical + SNPs (priority enforced)
clinical_cols = [
    c for c in train.columns
    if c not in SNP_COLS_ALL + [ID_COL, TARGET_SEVERITY, TARGET_MACE]
]

mace_snps = filter_by_maf(train, SNP_COLS_ALL, maf_threshold=0.01)
mace_snps = chi2_select_snps(train, TARGET_MACE, mace_snps, p_threshold=0.05)
mace_snps = enforce_priority(train, mace_snps, max_total=90)

mace_features = [AGE_COL, SEX_COL] + clinical_cols + mace_snps
mace_features = mace_features[:150]

# ===================================================
# MODELS (SCORING-AWARE)
# ===================================================

# ---- SEVERITY: optimize calibrated log-loss
sev_model = Pipeline([
    ("imputer", IterativeImputer(max_iter=15, random_state=42)),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.2,
        C=1.0,              # IMPORTANT: weak regularization
        max_iter=6000,
        random_state=42
    ))
])

# ---- MACE: optimize ordinal separation
mace_model = Pipeline([
    ("imputer", IterativeImputer(max_iter=15, random_state=42)),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.2,
        C=1.0,
        multi_class="multinomial",
        max_iter=7000,
        random_state=42
    ))
])

# ===================================================
# TRAIN
# ===================================================
print("Training models...")
sev_model.fit(train[sev_features], train[TARGET_SEVERITY])
mace_model.fit(train[mace_features], train[TARGET_MACE])

# ===================================================
# PREDICTION (CRITICAL PART)
# ===================================================

# ---- Severity: keep probabilities (log-loss)
severity_pred = sev_model.predict_proba(test[sev_features])[:, 1]

# ---- MACE: conservative ordinal post-processing
mace_proba = mace_model.predict_proba(test[mace_features])

mace_pred = np.zeros(len(mace_proba), dtype=int)
mace_pred[mace_proba[:, 2] > 0.45] = 2
mace_pred[
    (mace_proba[:, 1] > 0.35) &
    (mace_proba[:, 2] <= 0.45)
] = 1

# ===================================================
# SUBMISSION (STRICT FORMAT)
# ===================================================
submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET_MACE: mace_pred.astype(int),
    TARGET_SEVERITY: severity_pred.astype(float)
})

# VERY IMPORTANT: match dummy exactly
submission = submission[dummy.columns]

submission.to_csv("submission_u.csv", index=False)
print("✅ submission_u.csv generated")

