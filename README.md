# Credit Card Fraud Detection

My solution to the [Credit Card Fraud Detection](https://platform.olimpiada-ai.ro/en/problems/344) problem on the Romanian AI Olympiad platform (MLCompete): detect fraudulent transactions in a heavily imbalanced dataset.

**Result:** F1 = 0.88 on the fraud class (test set), **9th place** on the leaderboard.

## The data

- 227,845 training transactions, of which only **394 are fraud (0.17%)**
- Features: `Time` (seconds since the first transaction), `V1`–`V28` (anonymized PCA components) and `Amount`
- Evaluation: F1 score on the fraud class. Accuracy is meaningless here: predicting "not fraud" for everything scores 99.8%.

## Approach

**Subtasks 1–2 (data analysis):** pandas filtering and aggregation, and a Mahalanobis distance computed with NumPy (mean vector, covariance matrix and its inverse, vectorized over all fraud transactions).

**Subtask 3 (classification):** gradient-boosted trees with XGBoost.

- **Validation:** stratified 5-fold cross-validation, so every fold keeps the same fraud rate. Out-of-fold probabilities give an honest prediction for every training transaction.
- **Threshold tuning:** F1 depends strongly on where the probabilities are cut. I chose the threshold that maximizes out-of-fold F1 (about 0.3) instead of the default 0.5.
- **Hyperparameter search:** a grid over tree depth and `min_child_weight`, written by hand around a `cv_score` function. All settings landed between 0.86 and 0.875 CV F1, close to the noise level of 394 frauds.

## What I learned

- **Boosting can be numerically unstable on extremely rare classes.** My first model predicted no fraud at all: the training loss oscillated, each tree overcorrecting the previous one. XGBoost starts from the fraud rate (0.17%), where the loss curvature is tiny, and leaf values are gradient ÷ curvature, so a single tree could jump every prediction to an extreme. Setting `max_delta_step=1`, which caps each tree's step, fixed it: the same configuration went from F1 0 to about 0.87 in cross-validation.
- **Check the model, not just the score.** Printing the median predicted probability of the training frauds exposed the broken model immediately.
- **Small pandas details matter:** column-name alignment silently turning a computation into zeros, and integer answers written as `0.0` in the submission file.

## Stack

Python, NumPy, pandas, scikit-learn (cross-validation, metrics), XGBoost
