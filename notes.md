## what is machine bias
- when a machine learning model, trained on historical data, ends up treating different groups of people unequally — even if no one explicitly programmed it to.
- The key insight that makes it different from "the code has a bug": the model isn't malfunctioning. It's doing exactly what it was built to do — find patterns in past data and apply them going forward. The bias comes from the data and the world the data came from, not from broken code. If policing was uneven in the past, or if a label like "arrested" doesn't cleanly mean "guilty," the model faithfully learns and repeats that unevenness — often while looking completely accurate on average.

## what do you mean by auditing instrument?
- Two different jobs a model can do, and your project is doing the second one, not the first.

**Predicting** — you build a model to make a *new* decision. "Will this specific person reoffend?" The model's output is the thing you actually use.

**Auditing** — you build a model *not* to make decisions with, but to investigate whether an existing system (or the data itself) has a pattern in it you want to expose. The model's output isn't the point — what you learn by picking it apart afterward is the point.

In your project: you're not building `ghost-variable` so a judge can use it to score real defendants. You're building it, then breaking its predictions apart by race, to answer a question — "does bias creep in even when race isn't an input?" The model is the tool you use to ask that question, the same way a thermometer isn't the point of a science experiment, it's how you find out if something's actually hot.

That's why race is excluded from training but still used afterward — you're not trying to make the best possible predictor, you're using the predictor as an instrument to detect something hidden in the data.

## cost function for logisitic regression
Quick recap of MSE first, so the comparison makes sense. In linear regression, cost was:

`*cost = average of (prediction - actual)²*`

Squaring the error and averaging — simple, and it works fine there because predictions and actual values are both just plain numbers (like house prices).

Why this breaks for logistic regression:

Your predictions here aren't numbers like "300000" — they're probabilities, squished between 0 and 1 by sigmoid. If you tried to use MSE with these squished values, you get a technical problem: the cost function's shape becomes bumpy (mathematicians call this "non-convex") — it has multiple valleys instead of one smooth bowl. Gradient descent (which works by always walking downhill) can get stuck in a shallow fake valley and never find the actual best answer. This is a real, provable mathematical issue caused by squashing everything through sigmoid — not something we need to prove ourselves, just know it's why MSE is the wrong tool here.

The fix: log loss (also called binary cross-entropy). Different formula, one smooth bowl-shaped valley, always findable by gradient descent.

Here's the core idea, in plain language before the formula:

If the actual answer is 1 (this person did reoffend), and the model predicted a probability close to 1 (confident, correct) → cost should be very low.
If the actual answer is 1, but the model predicted a probability close to 0 (confident, wrong) → cost should be very high (a harsh penalty for confident wrongness).
Same logic mirrored for actual answer 0.

The formula, one example at a time:

`if y = 1:  cost = -log(prediction)`
`if y = 0:  cost = -log(1 - prediction)`

Why -log? Because log of a number close to 1 is close to 0 (low cost, good), and log of a number close to 0 is a huge negative number, so the - flips it to a huge positive cost (bad prediction = big penalty). This is designed specifically to punish confident wrongness harshly.

Combined into a single formula (this trick just merges both cases into one line, using the fact that whichever term isn't relevant gets multiplied by 0):

`*cost = -[ y * log(prediction) + (1-y) * log(1-prediction) ]*`

Walk through it with y=1: the second term (1-y) becomes (1-1)=0, killing that whole term, leaving just -log(prediction) — matches the first case above. With y=0: the first term y becomes 0, leaving -log(1-prediction) — matches the second case.

**Averaged over all your training examples, this is your total cost function.**

`J(w,b)=−1/m(∑m​[y(i)log(fw,b​(x(i)))+(1−y(i))log(1−fw,b​(x(i)))])`
Where:
- m = number of training examples
- y⁽ⁱ⁾ = the true label (0 or 1) for the i-th example
- f_{w,b}(x⁽ⁱ⁾) = the model's predicted probability for the i-th example, i.e. sigmoid(w·x⁽ⁱ⁾ + b)
- The 1/m and the sum: you're computing this cost for every single training example, then averaging across all of them — one overall number representing "how wrong is the model right now, on average, across the whole dataset"
- limit is form i = 1 to i = m

## what is y
- y is your label (also called "target" or "ground truth") — the actual correct answer for each person, not something the model computes, something you already know from the data.

- In your project specifically: y = two_year_recid — for each person, was 1 (they were rearrested within 2 years) or 0 (they weren't). This is real, historical, already-known fact about each person — it's what actually happened, pulled straight from the dataset.

- Why the cost function needs it: the whole point of "cost" is measuring "how wrong is the model." Wrong compared to what? Compared to y — the real answer. The model produces predictions (a guess, a probability between 0 and 1). y is the truth. The cost function compares them and produces a number saying how far off the guess was.


# Logistic Regression — From Scratch Notes

## Sigmoid

Squishes any real number into the range (0, 1) so raw linear output can be
interpreted as a probability.

    sigmoid(z) = 1 / (1 + e^(-z))

- sigmoid(0) = 0.5 (exact midpoint)
- large positive z → close to 1
- large negative z → close to 0

`np.exp(x)` = `e^x`, computed by NumPy instead of by hand.

---

## Why not use MSE (linear regression's cost function)?

MSE assumes predictions are plain numbers. Once predictions are squished
through sigmoid, using MSE makes the cost function's shape bumpy
(non-convex) — multiple valleys instead of one smooth bowl. Gradient
descent can get stuck in a fake valley and never find the true minimum.

## Cost function — binary cross-entropy (log loss)

One smooth bowl-shaped valley, always solvable by gradient descent.

For one example:
- if y = 1: cost = -log(prediction)
- if y = 0: cost = -log(1 - prediction)

Combined into one formula (only the relevant term survives, other is
multiplied by 0):

    cost = -[ y·log(prediction) + (1-y)·log(1-prediction) ]

Averaged over all m training examples — this is what gets implemented:

    J(w,b) = -(1/m) * sum( y*log(f(x)) + (1-y)*log(1-f(x)) )

Lower cost = better. Not meaningful as a single number on its own — only
meaningful compared to another cost value (e.g. before vs. after training).

---

## X.shape

`.shape` gives the dimensions of an array — (rows, columns).
`X.shape[0]` = number of rows = number of training examples (m).

## X @ w (matrix multiplication)

Computes `w1*x1 + w2*x2 + ... + wn*xn` for **every row at once**, no loop
needed. Returns one number per row (one prediction-input per person).

Example (3 people, 2 features):
    X = [[2,1],[0,0],[5,1]], w = [0.3, 0.6]
    X @ w = [1.2, 0, 2.1]

Then `+ b` adds bias to every row, and `sigmoid(...)` converts each into
a probability — all in one line, no loop.

## Why X.T (transpose) is needed in the gradient

X is shape (m, n_features). To get one gradient value *per feature*
(n_features numbers, not m numbers), the shapes need to line up as
(n_features, m) @ (m,) → (n_features,). Transpose just swaps rows and
columns to make that possible.

Proved by comparing against an explicit loop with the same math — both
methods gave identical results:
    dw_matrix = (1/m) * (X.T @ errors)
    # same answer as manually looping over each feature and each example
    # and summing (prediction[i]-y[i]) * X[i][j]

`X.T @ errors` **is** doing the sum from the formula — just via matrix
multiplication instead of a visible `np.sum` or loop.

---

## Gradient descent — mechanism

Repeat:
1. Compute current cost (how wrong the model is)
2. Compute the gradient (dw, db) — the direction that reduces cost
3. Nudge w and b a small step in that direction
4. Repeat until cost stops decreasing much (converged)

Update rule:
    w = w - alpha * dw
    b = b - alpha * db

alpha (learning rate) = how big a step to take each iteration.

Gradients (same structure as linear regression's gradient — only
"prediction" now means sigmoid(wx+b) instead of raw wx+b):

    dw = (1/m) * X.T @ (predictions - y)
    db = (1/m) * sum(predictions - y)

Verified on a toy 3-person example: cost dropped from 0.353 → 0.019 over
1000 iterations, confirming the implementation is correct.

---

## y (the label)

`y` = the real, historical, already-known answer for each row (ground
truth) — not something the model computes. In this project, y =
`two_year_recid` (did this person actually get rearrested within 2 years).
Cost compares `predictions` (model's guess) against `y` (the truth).

---

## Feature scaling (standardization)

Features on very different scales (e.g. priors_count: 0–30+ vs.
sex_Male: 0 or 1) slow down or destabilize gradient descent. Fix:

    x_scaled = (x - mean) / std

Only apply to real numeric features (priors_count) — not to already-
binary one-hot columns (sex_Male, age_cat_*, c_charge_degree_M), since
scaling those further doesn't help and distorts their meaning.

---

## pd.get_dummies (one-hot encoding) — mechanism

For a category column with N unique values:
1. Finds all N unique values
2. Creates N new columns, one per value, named `col_value`
3. For each row, puts 1 in the column matching that row's value, 0 in
   every other new column
4. `drop_first=True` removes one column (redundant — if all remaining
   category columns are 0, that row must be the dropped category)

Rule: N categories → N columns created → N-1 kept after drop_first.
Same mechanism regardless of how many categories there are (2, 3, 50...).
Caution: very high-cardinality columns (e.g. charge descriptions with
hundreds of unique values) shouldn't be one-hot encoded — too many
sparse columns, hurts training. This is why age_cat (3 buckets) was used
instead of raw age or c_charge_desc.

---

## Data filtering (ProPublica methodology)

Four filters applied, in order, on the raw 7214-row dataset:
1. `days_b_screening_arrest` between -30 and 30 (screening date and
   arrest date must be close together — otherwise the score likely
   isn't about this specific charge)
2. drop `is_recid == -1` (missing/unknown recidivism outcome)
3. drop `c_charge_degree == 'O'` (ordinary traffic offenses)
4. drop `score_text == 'N/A'` (no COMPAS score recorded)

Final: 7214 → 6172 rows. Matches ProPublica's reported filtered count.

Note: filters 2–4 removed 0 additional rows in this specific dataset
(verified by checking `.unique()` values and `.sum()` counts) — not
bugs, just how this version of the data happens to be structured.

---

## Base rate vs. error rate — the core fairness tension

A model can be *calibrated* (accurately reflecting real base-rate
differences between groups) and still have unequal false positive /
false negative rates between those groups — these are two different,
independent claims, and both can be true at once. This isn't a
contradiction — it's a mathematical consequence of groups having
different underlying base rates, not evidence the model is "wrong."
Arrest (the label used here) is also not identical to actual offending —
policing intensity varies by area/group, which is a real limitation.

## Auditing vs. predicting

Predicting: build a model to make a new decision on new data.
Auditing: build a model *not* to use for decisions, but to investigate
whether an existing system or the data itself contains a hidden pattern
(e.g. bias). The model's output isn't the point — what you learn by
breaking it apart afterward (e.g. by race) is the point.

## Threshold — converting probabilities to decisions

Sigmoid outputs a probability (0 to 1), not a decision. To audit the
model, real yes/no predictions are needed — standard threshold is 0.5.

    predicted_labels = (probabilities >= 0.5).astype(int)

`probabilities >= 0.5` gives a True/False array (same boolean-indexing
pattern as everywhere else) — True where probability crosses the
threshold. `.astype(int)` converts True→1, False→0. This is NOT
rounding to the nearest integer — every value above the threshold
becomes exactly 1, every value below becomes exactly 0, regardless of
how close to 0.5 it was (0.51 and 0.99 are treated identically).

---

## Fairness audit — FPR / FNR / accuracy by race, implementation notes

`race` was dropped when building X (never used as a model input) but
kept separately from the original df for auditing predictions after
the fact.

Boolean indexing pattern, extended to 3 stacked conditions with `&`:

    black_not_reoffend = (race == 'African-American') & (y == 0)
    black_fp = black_not_reoffend & (predicted_labels == 1)
    fpr_black = black_fp.sum() / black_not_reoffend.sum()

- Each condition alone gives True/False per person.
- `&` requires all conditions true simultaneously (same as the
  screening-window filter in data_prep.py).
- `.sum()` on a True/False array counts how many are True (True=1,
  False=0 in NumPy) — so `black_not_reoffend.sum()` = total count of
  Black defendants who didn't reoffend; `black_fp.sum()` = how many of
  those were false positives.
- Dividing gives the rate: FPR = false positives ÷ everyone eligible to
  be a false positive.

Same structure mirrored for FNR, swapping `y == 0` → `y == 1` and
`predicted_labels == 1` → `predicted_labels == 0`.

## Accuracy by race — array indexing with a boolean mask

    black_acc = (predicted_labels[race == 'African-American'] == y[race == 'African-American']).mean()

- `predicted_labels[race == 'African-American']` — filters the
  predictions array down to only Black defendants (same mask-indexing
  idea as `df[condition]`, just applied to a NumPy array).
- `y[race == 'African-American']` — same filter applied to the real
  outcomes, so both arrays stay aligned person-for-person.
- Comparing the two filtered arrays element-by-element gives True where
  the prediction matched reality.
- `.mean()` = `.sum() / count` = fraction of True values = accuracy.

---

## dtype bug — object arrays and np.exp

Building X from a DataFrame with mixed types (decimals + True/False)
produced a NumPy array with dtype `object` instead of a proper numeric
type. `np.exp` (inside sigmoid) failed on object arrays with a
confusing error (`'float' object has no attribute 'exp'`).

Fix: `X = X.to_numpy().astype(float)` — forces every value into proper
float64, converting True/False into 1.0/0.0 along the way. Diagnosed by
checking `X.dtype` before assuming the cause.

---

## Race sample sizes — why only Black vs. white were compared

    African-American    3175
    Caucasian            2103
    Hispanic              509
    Other                 343
    Asian                  31
    Native American        11

Black and white defendants make up ~87% of the dataset combined and are
large enough for statistically reliable FPR/FNR estimates. Smaller
groups (especially Native American, n=11) are too small — a single
misclassification would swing the rate by 10%+, making any computed
rate unreliable rather than meaningful. This mirrors why ProPublica's
own analysis centers on the Black-white comparison — not an arbitrary
choice, a sample-size constraint.

---

## Real results (9000 iterations, alpha=0.1)

| Metric | Black | White |
|---|---|---|
| FPR | 32.96% | 17.17% |
| FNR | 33.35% | 59.98% |
| Accuracy | 66.83% | 66.10% |

Overall accuracy is nearly identical between groups (~0.7% gap) while
FPR and FNR differ by ~1.9x and ~1.8x respectively, in opposite
directions. This is the central finding: aggregate accuracy conceals
the disparity entirely — it's only visible once the confusion matrix
is broken apart by subgroup. Directly reproduces ProPublica's original
2016 finding, from a model that never received race as an input
feature — meaning the disparity is being carried through a proxy
(most likely priors_count, per the base-rate/policing discussion
above).

sklearn validation: from-scratch weights matched sklearn's
LogisticRegression coefficients in sign and closely in magnitude
across all 5 features (e.g. priors_count: 0.805 vs 0.806), confirming
correct implementation.