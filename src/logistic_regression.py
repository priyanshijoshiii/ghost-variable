import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# Model functions 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, w, b):
    m = X.shape[0]
    z = X @ w + b
    predictions = sigmoid(z)
    cost = -(1/m) * np.sum(y * np.log(predictions) + (1-y) * np.log(1-predictions))
    return cost


def gradient_descent(X, y, alpha, w, b, num_iterations):
    m = X.shape[0]
    for i in range(num_iterations):
        z = X @ w + b
        predictions = sigmoid(z)
        dw = (1/m) * (X.T @ (predictions - y))
        db = (1/m) * np.sum(predictions - y)
        w = w - alpha * dw
        b = b - alpha * db
    return w, b


# ── Load and prepare real data ───────────────────

df = pd.read_csv('data/processed/compas_filtered.csv')

X = df[['priors_count', 'sex_Male', 'age_cat_Greater than 45', 'age_cat_Less than 25', 'c_charge_degree_M']]
y = df['two_year_recid']

X = X.to_numpy().astype(float)
y = y.to_numpy()

# scale priors_count (column 0) — only real numeric feature, rest are already 0/1
mean = X[:, 0].mean()
std = X[:, 0].std()
X[:, 0] = (X[:, 0] - mean) / std


#  Train from-scratch model 

w_init = np.zeros(X.shape[1])
b_init = 0

print("Cost before training:", compute_cost(X, y, w_init, b_init))

w_final, b_final = gradient_descent(X, y, alpha=0.1, w=w_init, b=b_init, num_iterations=9000)

print("Cost after training:", compute_cost(X, y, w_final, b_final))
print("Final w:", w_final)
print("Final b:", b_final)


# ── Validate against sklearn ──────────────────────

sklearn_model = LogisticRegression()
sklearn_model.fit(X, y)

print("sklearn coefficients:", sklearn_model.coef_)
print("sklearn intercept:", sklearn_model.intercept_)


# ── Generate predictions ──────────────────────────

z = X @ w_final + b_final
probabilities = sigmoid(z)
predicted_labels = (probabilities >= 0.5).astype(int)


#  Fairness audit: FPR / FNR / accuracy by race 

race = df['race'].to_numpy()

# FPR: among people who did NOT reoffend, what fraction were wrongly predicted high-risk
black_not_reoffend = (race == 'African-American') & (y == 0)
black_fp = black_not_reoffend & (predicted_labels == 1)
fpr_black = black_fp.sum() / black_not_reoffend.sum()

white_not_reoffend = (race == 'Caucasian') & (y == 0)
white_fp = white_not_reoffend & (predicted_labels == 1)
fpr_white = white_fp.sum() / white_not_reoffend.sum()

print("FPR (Black):", fpr_black)
print("FPR (White):", fpr_white)

# FNR: among people who DID reoffend, what fraction were wrongly predicted low-risk
black_reoffended = (race == 'African-American') & (y == 1)
black_fn = black_reoffended & (predicted_labels == 0)
fnr_black = black_fn.sum() / black_reoffended.sum()

white_reoffended = (race == 'Caucasian') & (y == 1)
white_fn = white_reoffended & (predicted_labels == 0)
fnr_white = white_fn.sum() / white_reoffended.sum()

print("FNR (Black):", fnr_black)
print("FNR (White):", fnr_white)

# Overall accuracy by race — shown to demonstrate it conceals the FPR/FNR disparity above
black_acc = (predicted_labels[race == 'African-American'] == y[race == 'African-American']).mean()
white_acc = (predicted_labels[race == 'Caucasian'] == y[race == 'Caucasian']).mean()

print("Accuracy (Black):", black_acc)
print("Accuracy (White):", white_acc)

print(df['race'].value_counts())