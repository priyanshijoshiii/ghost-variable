# Auditing Recidivism Risk Scores: A From-Scratch Replication of the ProPublica COMPAS Analysis

> An independent, from-scratch logistic regression audit of racial disparities in the COMPAS recidivism risk-scoring algorithm, replicating and extending the methodology of ProPublica's 2016 "Machine Bias" investigation.

---

## Table of Contents

- [Overview](#overview)
- [Background: What COMPAS Is](#background-what-compas-is)
- [Research Question](#research-question)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Filtering](#data-filtering)
  - [Features](#features)
  - [Models](#models)
  - [Fairness Metrics](#fairness-metrics)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Results](#results)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [What I Learned](#what-i-learned)
- [References](#references)
- [License](#license)

---

## Overview

This project implements logistic regression from scratch in NumPy and uses it to audit the COMPAS recidivism dataset for racial disparity, replicating and extending the methodology of ProPublica's 2016 "Machine Bias" investigation. Rather than training a model to predict an outcome, the model here is used as an auditing instrument: race is deliberately excluded from the feature set, and the resulting predictions are broken apart by race after the fact to test whether disparity emerges regardless.

The COMPAS dataset is one of the most widely cited datasets in the algorithmic fairness literature. This project engages with an active, unresolved question in that field: whether a classifier can simultaneously satisfy multiple competing definitions of fairness (equal error rates vs. predictive parity) when base rates differ across groups — and what that implies for any system making decisions from historical data.

This is not a production risk-assessment tool, and it does not propose a bias-mitigation method. It is a rigorous, from-scratch replication built to demonstrate an accurate understanding of classification, evaluation metrics under class imbalance, and the mathematics of fairness trade-offs.

---

## Background: What COMPAS Is

COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) is a proprietary risk-assessment tool developed by Northpointe (now Equivant). It produces a risk score from 1–10 estimating the likelihood that a defendant will reoffend within two years. The score has been used by judges and parole boards across the United States to inform decisions about bail, sentencing, and parole.

The model's internal logic was never publicly disclosed. In 2016, ProPublica journalists Julia Angwin, Jeff Larson, Surya Mattu, and Lauren Kirchner obtained COMPAS scores and criminal records for over 7,000 defendants in Broward County, Florida, and analyzed whether the algorithm's *errors* were evenly distributed across race — regardless of what was happening inside the black box.

Their central finding: Black defendants who did *not* reoffend were nearly twice as likely to be misclassified as high-risk compared to white defendants who did not reoffend. White defendants who *did* reoffend were more often mislabeled as low-risk. Overall accuracy between racial groups was similar — the disparity only appeared once the confusion matrix was broken apart by subgroup.

Northpointe disputed the framing, pointing out that COMPAS was well **calibrated** — within a given risk category, real-world reoffense rates were similar across race. Both claims turned out to be mathematically true simultaneously, which is itself the deeper finding formalized later by Chouldechova (2017) and Kleinberg et al. (2016): when base rates of an outcome differ between groups, no classifier can satisfy predictive parity, equal false-positive rates, and equal false-negative rates all at once. It is a mathematical impossibility, not an engineering failure.

---

## Research Question

> When a logistic regression model is trained to predict recidivism — with race deliberately excluded as an input feature — does it still reproduce racial disparities in false positive and false negative rates? And if so, why, given that race was never explicitly used?

Secondary questions this project investigates:
- Which single feature carries the most predictive weight, and does it correlate with race even though race itself is excluded (a proxy variable effect)?
- Does the model's own calibration (predictive parity) hold across race even while its error rates diverge?
- How do the findings from an independently trained model compare to ProPublica's audit of the actual proprietary COMPAS scores?

---

## Dataset

**Source:** [ProPublica's `compas-analysis` GitHub repository](https://github.com/propublica/compas-analysis)
**File used:** `compas-scores-two-years.csv`
**Population:** ~7,214 pretrial defendants assessed in Broward County, Florida, 2013–2014, with two years of follow-up criminal history.

Each row includes demographic information, criminal history, the COMPAS-assigned risk score and decile, and whether the person was rearrested within two years (`two_year_recid`).

This dataset is public and has been used in dozens of peer-reviewed fairness papers; it is not scraped or unofficially sourced.

---

## Methodology

This project deliberately mirrors ProPublica's own methodology page ("How We Analyzed the COMPAS Recidivism Algorithm") as closely as possible before extending it, so that results can be sanity-checked against a known, published baseline.

### Data Filtering

Following the original methodology exactly, rows are excluded if:

| Filter | Reason |
|---|---|
| `days_b_screening_arrest` not in [-30, 30] | Charge and COMPAS screening date too far apart to be plausibly related |
| `is_recid == -1` | No recidivism outcome could be determined |
| `c_charge_degree == 'O'` | Ordinary traffic offenses, not the offense class COMPAS is meant to score |
| `score_text == 'N/A'` | No COMPAS score was recorded for this defendant |

### Features

**Model inputs (used to train the classifier):**
- `age_cat` (categorical: Less than 25 / 25–45 / Greater than 45)
- `sex`
- `priors_count` (number of prior offenses)
- `c_charge_degree` (felony / misdemeanor)

**Explicitly excluded from model inputs:**
- `race` — held out entirely from training. Used only afterward, to slice predictions and audit for disparity. This mirrors the real-world claim that COMPAS does not use race directly, and tests whether bias can emerge anyway through correlated features.

**Label:**
- `two_year_recid` — ground truth: was this person rearrested within two years.

### Models

1. **Logistic Regression — implemented from scratch in NumPy.**
   - Sigmoid hypothesis, binary cross-entropy cost function, batch gradient descent, manual feature scaling (standardization), and L2 regularization implemented by hand — no scikit-learn in the core training loop.
   - Validated against `sklearn.linear_model.LogisticRegression` on the same train/test split to confirm the from-scratch implementation converges to equivalent parameters and predictions.

2. **Baseline comparison: raw COMPAS decile scores**, thresholded into Low/Medium/High exactly as in the original study, used as a second "model" for comparison purposes — this lets the project compare *my* model's disparity against the *actual* COMPAS disparity ProPublica reported.

### Fairness Metrics

For each model, predictions are broken out by race (Black / white, matching the original study's primary comparison) and the following are computed:

- **False Positive Rate (FPR):** among people who did *not* reoffend, what fraction were predicted high-risk.
- **False Negative Rate (FNR):** among people who *did* reoffend, what fraction were predicted low-risk.
- **Predictive Parity / Calibration:** within a given predicted-risk bucket, is the actual reoffense rate similar across race.
- **Odds ratio via auxiliary logistic regression:** a second regression of predicted risk category on race + controls (age, sex, priors), to test whether race remains statistically predictive of the score after controlling for other factors — directly mirroring the regression run in the original ProPublica analysis.

---

## Project Structure

```
compas-fairness-audit/
├── data/
│   ├── raw/                     # Original compas-scores-two-years.csv (not committed — see Setup)
│   └── processed/                # Filtered dataset after applying exclusion criteria
├── src/
│   ├── data_prep.py              # Filtering + feature engineering, mirrors methodology exactly
│   ├── logistic_regression.py    # From-scratch NumPy implementation (sigmoid, cost, gradient descent, regularization)
│   ├── evaluate.py                # Confusion matrices, FPR/FNR/calibration by subgroup
│   ├── audit.py                   # Race-on-score auxiliary regression, odds ratios
│   └── sklearn_baseline.py        # sklearn comparison model, used only for validation
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory analysis, score distributions by race
│   ├── 02_model_training.ipynb     # Training + convergence checks (scratch vs sklearn)
│   └── 03_fairness_audit.ipynb     # Full metric breakdown + visualizations
├── results/
│   ├── figures/                    # Saved plots (score distributions, ROC curves, FPR/FNR bar charts)
│   └── metrics_summary.md          # Final numeric results table
├── tests/
│   └── test_logistic_regression.py # Unit tests for the from-scratch model
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Tech Stack

| Purpose | Tool |
|---|---|
| Core model | Python 3.11, NumPy (from-scratch logistic regression) |
| Data handling | pandas |
| Validation baseline | scikit-learn |
| Visualization | matplotlib, seaborn |
| Statistical checks | scipy.stats |
| Notebooks | Jupyter |
| Testing | pytest |

No deep learning frameworks are used — intentionally. The core model is a from-scratch implementation of regularized logistic regression trained via batch gradient descent, so that every line of it is understood and explainable rather than imported as a black box.

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/compas-fairness-audit.git
cd compas-fairness-audit

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the dataset
curl -o data/raw/compas-scores-two-years.csv \
  https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
```

---

## Usage

```bash
# 1. Filter and prepare the dataset (mirrors ProPublica's exclusion criteria)
python src/data_prep.py

# 2. Train the from-scratch logistic regression model
python src/logistic_regression.py --epochs 5000 --lr 0.01 --l2 0.1

# 3. Validate against sklearn
python src/sklearn_baseline.py

# 4. Run the full fairness audit (FPR/FNR/calibration by race, odds ratio regression)
python src/audit.py
```

Or run the notebooks in order (`01_eda.ipynb` → `02_model_training.ipynb` → `03_fairness_audit.ipynb`) for an annotated, visual walkthrough.

---

## Results

*(To be filled in once training and audit are complete.)*

| Metric | Black defendants | White defendants |
|---|---|---|
| False Positive Rate | — | — |
| False Negative Rate | — | — |
| Predictive Parity (High-risk bucket) | — | — |
| Overall Accuracy | — | — |

| Model | Overall Accuracy | FPR Gap (Black − White) | FNR Gap (Black − White) |
|---|---|---|---|
| From-scratch logistic regression | — | — | — |
| Raw COMPAS score (baseline) | — | — | — |

---

## Key Findings

*(To be written after the audit — this section should state plainly which fairness definition the model satisfies, which it violates, and why it cannot satisfy both, per Chouldechova's impossibility result.)*

---

## Limitations

- This is a pedagogical replication, not a peer-reviewed research contribution. Sample sizes, feature engineering choices, and threshold definitions may differ subtly from ProPublica's original notebook.
- `race` is used only as an audit variable, never as a model feature — but proxy effects through `priors_count` and geography-correlated features are not fully disentangled here.
- Two-year arrest (not conviction) is used as the recidivism label, following the original study — this is a known point of criticism in the fairness literature, since arrest itself is not race-neutral.
- The project audits disparity; it does not attempt to "fix" it. Any mitigation technique (reweighing, threshold adjustment, adversarial debiasing) is out of scope for this iteration.

---

## What I Learned

*(Personal reflection section — to be completed after the project. Suggested prompts: What surprised you about the FPR/FNR gap? Did your prediction about proxy variables hold up? Which fairness definition do you now think matters most, and why?)*

---

## References

1. Larson, J., Mattu, S., Kirchner, L., & Angwin, J. (2016). *How We Analyzed the COMPAS Recidivism Algorithm.* ProPublica.
2. Angwin, J., Larson, J., Mattu, S., & Kirchner, L. (2016). *Machine Bias.* ProPublica.
3. Chouldechova, A. (2017). *Fair Prediction with Disparate Impact: A Study of Bias in Recidivism Prediction Instruments.* Big Data, 5(2).
4. Kleinberg, J., Mullainathan, S., & Raghavan, M. (2016). *Inherent Trade-Offs in the Fair Determination of Risk Scores.* arXiv:1609.05807.
5. Flores, A. W., Bechtel, K., & Lowenkamp, C. T. (2016). *False Positives, False Negatives, and False Analyses: A Rejoinder to "Machine Bias."* Federal Probation, 80(2).
6. ProPublica. `compas-analysis` [Dataset & original notebook]. GitHub.

---

## License

This project is released under the MIT License (see `LICENSE`). The COMPAS dataset itself is publicly released by ProPublica under its own terms — see the [compas-analysis repository](https://github.com/propublica/compas-analysis) for dataset licensing details.

---

*Independent research project by [Your Name].*
