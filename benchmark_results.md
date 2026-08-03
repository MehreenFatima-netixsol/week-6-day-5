# Match-Winner Model vs. Naive Benchmarks

Held-out set: **911 matches**, disjoint from model training data.

| Predictor | Accuracy |
|---|---|
| Always pick home team | 54.7% |
| Naive (better recent win-rate wins) | 65.6% |
| **Trained model (LogisticRegression)** | **67.3%** |

**Model lift over the naive form-based baseline: +1.6 points.**

Context: AFL match outcomes are inherently noisy (recent public modelling and bookmaker-implied win probabilities for the real competition typically sit in the low-to-mid 70s% accuracy range across a season). A few points of lift over a naive recent-form heuristic is a realistic, defensible result for a lightweight logistic model on this feature set -- it is not a large edge, and should be presented to stakeholders as directionally useful rather than betting-grade.