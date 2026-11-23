# Credit Scoring AI

[![Build and test](https://github.com/simohamedd/credit-scoring-ai/actions/workflows/build.yml/badge.svg)](https://github.com/simohamedd/credit-scoring-ai/actions/workflows/build.yml)

Decides whether to approve a loan using a decision tree written from scratch in Java (CART with Gini impurity). Every decision comes with the exact rules behind it, which matters in lending, where customers have a right to an explanation.

## How it works
1. **Data:** `SyntheticApplicants` creates past loans whose chance of repayment depends on income, debt-to-income, credit history, missed payments and loan size, through a hidden rule plus randomness.
2. **Model:** `DecisionTree` finds the split that best separates repaid from defaulted loans at each step, up to a maximum depth, with a minimum number of applicants per leaf to avoid overfitting.
3. **Tuning:** `CrossValidation` tries tree depths 1 to 7 with 5-fold cross-validation and picks the one that generalises best, instead of a hand-picked value.
4. **Decision:** `CreditScorer` turns the predicted repay chance into **Approve** (75% or more), **Refer** to a human (50–75%) or **Decline**.
5. **Explanation:** for each applicant, the rules followed, for example "missed payments is above 1.50 (applicant: 3)". When the tree splits on the same detail more than once on the way down, those splits are merged into one range, so each detail is mentioned once: "missed payments is above 2.50 and at most 3.50 (applicant: 3)" rather than three separate rules.
6. **Feature importance:** `DecisionTree.featureImportances()` ranks the applicant details by how much the model relies on them overall (Gini importance: the impurity each feature's splits remove, weighted by how many applicants reach them, as a share of the total). Shares sum to 1; a feature the tree never splits on scores 0.

## Input validation
Applications that cannot describe a real person are rejected with an `IllegalArgumentException` instead of being scored:
- annual income must be positive (the model uses loan / income, which is undefined at zero);
- debt-to-income must be between 0 and 1;
- credit history, missed payments and loan amount cannot be negative;
- missing (`NaN`) or infinite values are refused everywhere.

`DecisionTree.fit` likewise refuses an empty dataset, a label count that doesn't match the rows, rows of different widths, a feature-name count that doesn't match the columns, and missing or infinite values, naming the offending row. Once trained, `predictProbability` and `explain` only accept a row with exactly one finite value per training feature; a short, long or `NaN`-containing row is refused rather than quietly sent down the wrong branch.

`CrossValidation` needs a non-empty list, between 2 and *n* folds, and a depth range with 1 ≤ min ≤ max.

## Lending policy
The 75% / 50% cut-offs are only the default. A lender can apply its own risk appetite to the same trained model, without retraining:
```java
CreditScorer cautious = scorer.withThresholds(0.85, 0.6); // approve from 85%, refer from 60%
CreditScorer noReferrals = scorer.withThresholds(0.7, 0.7); // approve or decline, nothing referred
```
A repay chance exactly at a cut-off gets the higher outcome. Cut-offs must satisfy 0 ≤ refer ≤ approve ≤ 1; anything else (including `NaN`) is refused with an `IllegalArgumentException`. `approveAt()` and `referAt()` report the policy a scorer is using, and the original scorer is never changed.

To see what a policy does before using it, run it over applicants whose outcome is known:
```java
Map<CreditScorer.Outcome, CreditScorer.Band> report = cautious.policyReport(pastApplicants);
report.get(CreditScorer.Outcome.APPROVE).share();      // share of applicants approved
report.get(CreditScorer.Outcome.APPROVE).repaidRate(); // how many of those really repaid
```
Every outcome appears in the report, in approve, refer, decline order; a band nobody falls into has 0 applicants and a `NaN` repayment rate. `policyReport` and `accuracy` refuse an empty list.

## Save and load
Training (with cross-validation) only needs to happen once. A trained scorer, including its cut-offs, can be saved to a plain text file and loaded later:
```java
scorer.save(Path.of("credit-model.txt"));
CreditScorer same = CreditScorer.load(Path.of("credit-model.txt"));
```
The loaded scorer gives the same repay chances, outcomes, explanations and feature importances as the one that was saved. Loading refuses, with an `IOException` naming the line at fault, a file that is truncated, has extra content, contains invalid numbers (a repay chance outside 0–1, a split deeper than the tree's maximum depth), or was trained on different applicant details.

## Run
```bash
mvn -q compile exec:java
```
Prints the cross-validation scores for each depth, test accuracy against a simple baseline, the learned rules, the feature-importance ranking, how the default and a cautious policy split the test applicants, and two example decisions with reasons.

## Test
```bash
mvn test
```

## Licence
MIT. See [LICENSE](LICENSE).
