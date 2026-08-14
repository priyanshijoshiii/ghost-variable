## what is machine bias
- when a machine learning model, trained on historical data, ends up treating different groups of people unequally — even if no one explicitly programmed it to.
- The key insight that makes it different from "the code has a bug": the model isn't malfunctioning. It's doing exactly what it was built to do — find patterns in past data and apply them going forward. The bias comes from the data and the world the data came from, not from broken code. If policing was uneven in the past, or if a label like "arrested" doesn't cleanly mean "guilty," the model faithfully learns and repeats that unevenness — often while looking completely accurate on average.

## what do you mean by auditing instrument?
- Two different jobs a model can do, and your project is doing the second one, not the first.

**Predicting** — you build a model to make a *new* decision. "Will this specific person reoffend?" The model's output is the thing you actually use.

**Auditing** — you build a model *not* to make decisions with, but to investigate whether an existing system (or the data itself) has a pattern in it you want to expose. The model's output isn't the point — what you learn by picking it apart afterward is the point.

In your project: you're not building `ghost-variable` so a judge can use it to score real defendants. You're building it, then breaking its predictions apart by race, to answer a question — "does bias creep in even when race isn't an input?" The model is the tool you use to ask that question, the same way a thermometer isn't the point of a science experiment, it's how you find out if something's actually hot.

That's why race is excluded from training but still used afterward — you're not trying to make the best possible predictor, you're using the predictor as an instrument to detect something hidden in the data.