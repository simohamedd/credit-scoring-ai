# Credit Scoring AI

[![Build and test](https://github.com/simohamedd/credit-scoring-ai/actions/workflows/build.yml/badge.svg)](https://github.com/simohamedd/credit-scoring-ai/actions/workflows/build.yml)

Decides whether to approve a loan using a decision tree written from scratch in Java (CART with Gini impurity). Every decision comes with the exact rules behind it, which matters in lending, where customers have a right to an explanation.

## How it works
1. **Data:** `SyntheticApplicants` creates past loans whose chance of repayment depends on income, debt-to-income, credit history, missed payments and loan size, through a hidden rule plus randomness.
2. **Model:** `DecisionTree` finds the split that best separates repaid from defaulted loans at each step, up to a maximum depth, with a minimum number of applicants per leaf to avoid overfitting.
3. **Decision:** `CreditScorer` turns the predicted repay chance into **Approve** (75% or more), **Refer** to a human (50–75%) or **Decline**.
4. **Explanation:** for each applicant, the rules followed, for example "missed payments is above 1.5 (applicant: 3)".

## Run
```bash
mvn -q compile exec:java
```
Prints test accuracy against a simple baseline, the learned rules, and two example decisions with reasons.

## Test
```bash
mvn test
```

## Licence
MIT. See [LICENSE](LICENSE).
