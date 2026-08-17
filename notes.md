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
