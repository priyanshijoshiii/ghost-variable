# .shape tells you the dimensions of an array — how many rows and columns it has.
# X.shape[0] grabs just the first number from that pair
# z = X @ w + b computes the raw linear score for every person in your dataset in one line, no loop needed.
## print(sigmoid(0))     # returns 0.5
## print(sigmoid(-10))   #returns a number near 0
## print(sigmoid(10))    #returns a number near 0
# X_test = np.array([[2, 1], [0, 0], [5, 1]])
# y_test = np.array([1, 0, 1])
# w_test = np.array([0.3, 0.6])
# b_test = -0.1

# z = X_test @ w_test + b_test
# predictions = sigmoid(z)
# errors = predictions - y_test

# # Method 1: matrix multiplication (what we're using)
# dw_matrix = (1/3) * (X_test.T @ errors)
# print("Using X.T @ errors:", dw_matrix)

# # Method 2: explicit loop with sum, matching the formula literally
# dw_loop = np.zeros(2)
# for j in range(2):
#     total = 0
#     for i in range(3):
#         total += errors[i] * X_test[i][j]
#     dw_loop[j] = total / 3
# print("Using explicit loop:", dw_loop)
# X_test = np.array([[2, 1], [0, 0], [5, 1]])
# y_test = np.array([1, 0, 1])
# w_test = np.array([0.3, 0.6])
# b_test = -0.1

# print("Cost before training:", compute_cost(X_test, y_test, w_test, b_test))

# w_final, b_final = gradient_descent(X_test, y_test, alpha=0.1, w=w_test, b=b_test, num_iterations=1000)

# print("Cost after training:", compute_cost(X_test, y_test, w_final, b_final))
# print("Final w:", w_final)
# print("Final b:", b_final)

# what does probabilities >= 0.5 actually produce before .astype(int) touches it? Same pattern as your very first boolean-indexing lesson: it's a True/False array, one per person. E.g. [0.75, 0.3, 0.6] >= 0.5 → [True, False, True].
# Then .astype(int) converts each: True → 1, False → 0. So [True, False, True] becomes [1, 0, 1].

import numpy as np
import pandas as pd

# ── Model functions ──────────────────────────────

def sigmoid(z):
    return 1/(1+ np.exp(-z))


def compute_cost(X,y,w,b):
    m = X.shape[0]                 # calculate the number of training examples
    z = X@w +b                     #linear part, for all rows at once
    predictions = sigmoid(z)       # squish into possibilities
    cost = -(1/m) * (np.sum(y*np.log(predictions) + (1-y)*np.log(1-predictions))) 
    return cost
    


def gradient_descent(X, y, alpha, w, b, num_iterations):
    m = X.shape[0]
    for i in range(num_iterations):
        z = X@w +b
        predictions = sigmoid(z)
        dw = (1/m)* X.T @(predictions-y)
        db = (1/m) * np.sum(predictions -y)
        w = w - alpha*dw
        b = b- alpha*db
    return w, b    


# ── Load and prepare real data ───────────────────

df = pd.read_csv('data/processed/compas_filtered.csv')

X = df[['priors_count', 'sex_Male', 'age_cat_Greater than 45', 'age_cat_Less than 25', 'c_charge_degree_M']]
y = df['two_year_recid']

X = X.to_numpy()
X = X.astype(float)
y = y.to_numpy() 

print(X.dtype)

# scale priors_count (column 0) — only numeric feature, others are already 0/1
mean = X[:, 0].mean()
std = X[:, 0].std()
X[:,0] = (X[:,0] - mean)/std

print(X.shape)
print(y.shape)
print(X[:5])

w_init = np.zeros(X.shape[1])
b_init = 0

print("cost before training", compute_cost(X, y, w_init, b_init))

w_final, b_final = gradient_descent(X, y, alpha=0.1, w=w_init, b=b_init, num_iterations=1000)

print("cost after training", compute_cost(X, y, w_final, b_final))
print("final w: ", w_final)
print("final b: ", b_final)


from sklearn.linear_model import LogisticRegression
sklearn_model = LogisticRegression()
sklearn_model.fit(X,y)

print("sklearn coefficients: ", sklearn_model.coef_)   # w
print("sklearn intercept: ", sklearn_model.intercept_)  # b

