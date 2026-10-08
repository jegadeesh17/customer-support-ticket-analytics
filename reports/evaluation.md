# Support Ops Intelligence — Evaluation Report

## Methodology
- 80/20 stratified train-test split (classification tasks)
- Up to 30,000 row sample per task for laptop-friendly training
- Labels engineered from business heuristics — see README caveats

## Metrics

### Priority Classification
- **model:** Gradient Boosting
- **accuracy:** 0.821
- **target:** >= 0.80

### Resolution Regression
- **model:** Gradient Boosting
- **r2:** 0.7343
- **target:** >= 0.70

### Satisfaction Classification
- **model:** Gradient Boosting
- **accuracy:** 0.9332
- **target:** >= 0.75

## Limitations
- Metrics reflect pipeline validity on engineered labels, not human-annotated ground truth.
- Regenerate: `python src/train_models.py && python scripts/export_evaluation.py`
