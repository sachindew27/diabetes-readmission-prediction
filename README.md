# Predicting Diabetes-Related Hospital Readmission

Can a hospital encounter record predict readmission, using classifiers written from scratch
rather than pulled off the shelf?

## Question

Readmissions are expensive for hospitals and dangerous for patients. Given the record of a
diabetic patient's stay (diagnoses, procedures, medications, discharge disposition, prior
visits), can we predict whether that patient returns?

The assignment constraint shapes everything here: the classifiers are implemented in NumPy
from first principles, so the point is understanding the mechanics rather than maximizing a
leaderboard score.

## Data

UCI dataset 296, "Diabetes 130-US hospitals 1999-2008", loaded via `ucimlrepo`: 101,766
encounters, 45 columns. Dropped `weight`, `payer_code` and `medical_specialty` (heavily
missing), and rows with null diagnosis codes, leaving 100,244 encounters.

Diagnoses are ICD-9. The first three digits give the disease category; the notebook keeps
the top 50 values per diagnosis column (about 75% of records) and buckets the tail as
"Others". Without that step one-hot encoding produced 2,288 columns; after it, 174.

**Target:** the raw label has three levels (`NO`, `<30`, `>30`). This binarizes to any
readmission, which is broader than the 30-day problem the UCI abstract describes. Resulting
balance is 53.69% not readmitted, so **53.69% is the baseline** every model must beat.

## Approach

Diagnostics came first: distribution and Q-Q plots to decide which features are near-Gaussian
(needed to pick a PDF or a PMF per feature in Naive Bayes), hypothesis tests per feature, and
pairplots on raw, polynomial and log-transformed features to test linear separability. Only
`number_inpatient` separated the classes, which is why the logistic regression feature set is
deliberately narrow.

Three models: **Naive Bayes from scratch** (detects distribution type per feature, switches
between PDF and PMF, Laplace smoothing), **logistic regression from scratch** (gradient
descent on the sigmoid, learning rate 0.01, up to 5,000 iterations), and a scikit-learn
`MLPClassifier` tuned with `RandomizedSearchCV`. All compared on accuracy, precision,
sensitivity, F1, specificity, plus a bias-variance decomposition and runtime and memory.

## Results

Against a 53.69% majority-class baseline:

| Model | Accuracy | Precision | Sensitivity | F1 | Specificity |
|---|---|---|---|---|---|
| Neural network | 63.07% | 61.75% | 54.12% | 57.68% | 70.86% |
| Naive Bayes (scratch) | 61.07% | 60.80% | 45.81% | 52.26% | 74.33% |
| Logistic regression (scratch) | 52.16% | 49.25% | 94.62% | 64.78% | n/a |

**Honest reading.** The neural network wins but beats the trivial baseline by only about 9
points. The from-scratch logistic regression lands *below* the baseline: it collapses toward
predicting readmission for nearly everyone, which is why its sensitivity and F1 look high
while its accuracy does not. That is the real result, reported rather than hidden. This
feature set with these models does not predict readmission well, which matches the published
literature on this dataset.

## Running it

Open the notebook in Jupyter or Colab. The dataset downloads automatically via `ucimlrepo`
(id 296). Requires pandas, NumPy, SciPy, scikit-learn, mlxtend, memory-profiler. The
full-data Naive Bayes fit takes roughly 17 minutes.

## Context

Graduate coursework, IE 7300 Statistical Learning for Engineering, Northeastern University,
October 2023. Four-person team project. The assignment required implementing the algorithms
from scratch, which is why hand-rolled classifiers sit next to a scikit-learn neural network.
