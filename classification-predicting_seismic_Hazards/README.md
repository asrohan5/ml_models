# Predicting Seismic Hazards in a Coal Mine
## Classification — What Works, What Breaks, and Why

---

### Business Problem

A coal mining operation monitors seismic activity every 8-hour shift using geophones and acoustic sensors. The question: **given sensor readings from this shift, will a high-energy seismic bump (rockburst, >10,000 Joules) occur in the next shift?**

If the model predicts hazard:
- Safety team deploys distress-shooting to reduce rock tension, or
- Workers are evacuated from the threatened zone

**Cost asymmetry:** A missed hazard can kill miners. A false alarm costs one 8-hour work stoppage. These are NOT equal costs. This asymmetry drives every metric and threshold decision in this notebook.


---

### Experiments Overview



| Experiment | Wrong approach | What breaks | Right approach |
|-----------|----------------|-------------|----------------|
| 1+2 — Metric | Accuracy | 93.4% with zero hazards caught | Recall, F-beta, PR metrics |
| 2 — Baseline | Skip DummyClassifier | No reference point | Always run dummy first |
| 3 — Confusion matrix | Just report accuracy | Miss the business cost of each cell | Translate each cell to business consequence |
| 4+5 — F1 vs F-beta | Always use F1 | Equal weighting when costs are asymmetric | F2 when recall > precision matters |
| 6 — ROC vs PR | Always use ROC-AUC | Looks great on imbalanced data while model is useless | Use PR + Average Precision for rare classes |
| 7 — Threshold | Default 0.5 everywhere | Assumes equal cost — wrong for this problem | Tune on CV predictions, report as a business choice |
| 7 — Threshold leakage | Tune on test set | Exploits test knowledge, won't hold in production | Tune only on CV out-of-fold predictions |
| 8 — CV predictions | .predict(X_train) | Seen data, metrics inflated | cross_val_predict for honest estimates |
| 9 — KFold | Regular KFold | Folds can have near-zero positive class | StratifiedKFold always for imbalanced data |
| 10 — Scaling | Skip for all models | SGD/LR/SVM degrade on multi-magnitude features | Scale for gradient/distance models; not needed for trees |

---


Every wrong choice in classification comes from the same root: **treating all errors as equal when the business problem says they are not.** Accuracy treats every prediction the same. F1 treats FP and FN the same. ROC-AUC treats a mistake on a common class the same as a mistake on a rare one. The metric must reflect the business cost — not just what is easy to compute or looks impressive in a report.

