## Documentation: CardiHack Prediction Models

# LMDM-m1geniomhe-group-8

🫀 CardI-HACK — Project Status
📌 Project Objective

This documentation explains the overall workflow of our first predictive model taking into consideration multiple approaches: the significant priority SNPs based on chi square test, different machine learning clustering approaches and logistic regression models. This is to test the model under different approaches for a high scoring and an accurate model.

📊 Data Description (Current Understanding)

- Dataset size: 347 patients data
- Data origin: obtained from several hundred real patients
- Data types:
    - Baseline clinical data (demographics, cardiac measurements, medical history)
    - Genetic data:
      - Mendelian variant (Variant.Pathogene)
      - 288 SNPs (75 priority SNPs)

Outcomes
- OUTCOME_SEVERITY (binary: 0 = non-severe, 1 = severe)
- OUTCOME_MACE (multiclass: 0 = no event, 1 = event ≤1000 days, 2 = event ≤3000 days)
Missing values are present and must be handled by all models.

🔬 Tested Workflows (So Far)
  1️⃣ Preprocessing
- Basic data inspection
- Handling of categorical and binary variables
- No advanced imputation strategy implemented yet (baseline handling only)

2️⃣ Feature Analysis & Selection
  ✅ Chi-Square Tests
- Chi-square tests of independence were performed:
  - For Outcome Severity
  - For Outcome MACE

- Purpose:
  - Identify SNPs statistically associated with each outcome
  - Provide an initial ranking of potentially informative genomic variables

- Result:
  - Several clinical variables and SNPs show significant associations
  - These results guide future feature selection steps

⚠️ At this stage, chi-square tests are exploratory only and not yet integrated into a final predictive pipeline.

3️⃣ Unsupervised Learning
  ✅ K-Means Clustering
- K-Means clustering applied to baseline patient data
- Objective:
  - Explore natural groupings of patients
  - Assess whether clusters align with outcomes (Severity / MACE)
- Observations:
  - Patient clusters show partial differentiation in clinical characteristics
  - No strong one-to-one correspondence with outcomes yet

✅ Other Clustering Algorithms
- Additional clustering methods tested (exploratory)
- Used to compare cluster stability and interpretability
- Clustering remains descriptive, not predictive at this stage

4️⃣ Supervised Learning
  ✅ Decision Tree
- A Decision Tree classifier was implemented as a first supervised model
- Applied to:
  - Outcome Severity
  - Outcome MACE (exploratory)
- Motivation:
  - Interpretability
  - Ability to handle mixed data types

- Current usage:
  - Baseline model
  - Used mainly to understand feature splits and decision rules

⚠️ No hyperparameter optimization or feature constraints enforced yet (e.g., SNP limits).

5️⃣ Data Augmentation
  ❌ Not implemented yet

6️⃣ Postprocessing
  ❌ Not implemented yet
  - No threshold optimization or calibration performed

📈 Results (Current)
  - Platform submission: ❌ Not submitted yet
  - Official score: Not available at this stage
The current phase focuses on exploration and methodological validation, not performance optimization.

🧠 Preliminary Analysis
- Chi-square tests suggest meaningful associations between certain clinical/genetic features and both outcomes.
- Unsupervised clustering highlights heterogeneity among patients but does not yet yield clear outcome-driven groups.
- Decision trees provide interpretable decision paths but are likely insufficient alone for optimal performance.
- Genetic data (SNPs) will require:
  - Careful feature selection
  - Dimensionality reduction
  - Strict adherence to challenge constraints (≤100 SNPs, ≥1 priority SNP for MACE)

🚧 Current Limitations
- No advanced missing data handling
- No feature selection pipeline integrated into modeling
- No model comparison or cross-validation
- No performance metrics reported yet
- Genetic data underutilized in current models

🔜 Next Steps
- Implement robust missing data strategies
- Integrate chi-square results into feature selection
- Enforce SNP constraints for MACE prediction
- Test ensemble models (Random Forest, Gradient Boosting)
- Perform cross-validation
- Submit first model to platform and report scores

📌 Status Summary

Project status: 🟡 Exploratory / Early Modeling Phase
Focus on data understanding, statistical testing, clustering, and baseline models.
