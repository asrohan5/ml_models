import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from scipy.io import arff
from io import BytesIO, StringIO
import urllib.request

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold,
    cross_val_score, cross_val_predict
)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, fbeta_score,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)



np.random.seed(42)
#print('Setup Complete')


#------------------------------------------------------------------------------------------------
#SECTION 1 - LOAD AND UNDERSTAND DATA


URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff'

with urllib.request.urlopen(URL) as resp:
    text_data = resp.read().decode('utf-8')
    raw_data, meta = arff.loadarff(StringIO(text_data))

df = pd.DataFrame(raw_data)

# ARFF loads nominal values as byte strings - decode them
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.decode('utf-8')

df = df.rename(columns={'class':'hazard'})
df['hazard'] = df['hazard'].astype(int)

#print(df.head())

counts = df['hazard'].value_counts()

overview_df = pd.DataFrame({
    'shape': df.shape,
    'missing_values': df.isnull().sum().sum(),
    'safe_shifts': counts[0],
    'safe_shift_pct': (counts[0]/len(df))*100,
    'hazard_shifts': counts[1],
    'hazard_shifts_pct': (counts[1]/len(df))*100,
    'imbalance_ratio': counts[0]/counts[1]

})

#print(overview_df)

CATEGORICAL = ['seismic', 'seismoacoustic', 'shift', 'ghazard']
NUMERIC = [
    'genergy', 'gpuls', 'gdenergy', 'gdpuls',
    'nbumps', 'nbumps2', 'nbumps3', 'nbumps4', 'nbumps5',
    'nbumps6', 'nbumps7', 'nbumps89', 'energy', 'maxenergy'
]

# print('Categorical feature values:')
# for col in CATEGORICAL:
#     print(f'  {col}: {sorted(df[col].unique())}')


#Energy features — scale range
# for col in ['genergy', 'energy', 'maxenergy']:
#     print(f'  {col}: {df[col].min():.0f} to {df[col].max():,.0f}')

#Energy spans 10^4 to 10^8 — a 10,000x range
#This will dominate gradient-based models without scaling


X = df.drop('hazard', axis=1)
y = df['hazard']

#stratifying to ensure the target classes which are very less compratively are equally distributed
#without which there are chances for all 'y' to fall in training and none in test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# Build preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), NUMERIC),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
], remainder='drop')

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep  = preprocessor.transform(X_test)
#print(f'\nFeature matrix shape after preprocessing: {X_train_prep.shape}')


#------------------------------------------------------------------------------------
#Building Models

def get_model_scores(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    model_preds = model.predict(X_test)
    model_acc = accuracy_score(y_test, model_preds)
    model_prec = precision_score(y_test, model_preds, zero_division=0)
    model_rec = recall_score(y_test, model_preds, zero_division=0)
    model_cm = confusion_matrix(y_test, model_preds)
    
    return model_preds, model_acc, model_prec, model_rec, model_cm

sgd_clf = SGDClassifier(loss='hinge', class_weight='balanced', random_state=42, max_iter=1000)

lr_clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

dummy_maj  = DummyClassifier(strategy='most_frequent').fit(X_train_prep, y_train)

dummy_rand = DummyClassifier(strategy='stratified', random_state=42).fit(X_train_prep, y_train)



sgd_preds, sgd_acc, sgd_prec, sgd_rec, sgd_cm = get_model_scores(sgd_clf, X_train_prep, y_train, X_test_prep, y_test)
lr_preds, lr_acc, lr_prec, lr_rec, lr_cm = get_model_scores(lr_clf, X_train_prep, y_train, X_test_prep, y_test)
dmaj_preds, dmaj_acc, dmaj_prec, dmaj_rec, dmaj_cm = get_model_scores(dummy_maj, X_train_prep, y_train, X_test_prep, y_test)
drand_preds, drand_acc, drand_prec, drand_rec, drand_cm = get_model_scores(dummy_rand, X_train_prep, y_train, X_test_prep, y_test)


#The dummy majority model matches or beats most naive models on accuracy.
#Accuracy is NOT the right metric. Recall is the primary concern here.
#A model that reports "93% accuracy" in this context should immediately raise red flags.


lr_tn, lr_fp, lr_fn, lr_tp = lr_cm.ravel()

#TN: Safe shifts correctly called safe -> normal operations continue
#FP: Safe shifts wrongly flagged hazard -> unnecessary evacuation, 8hr production loss
#FN: Hazardous shifts missed -> workers in rockburst zone — life risk
#TP: Hazardous shifts correctly caught -> countermeasures deployed, lives saved

#Precision = TP/(TP+FP)
lr_prec = lr_tp/(lr_tp + lr_fp) 
#Recall = TP/(TP+FN) 
lr_rec = lr_tp/(lr_tp + lr_fn)



#-------------------------------------------------------------------------------------------------------------------------------
#F Scores

rf_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

rf_preds, rf_acc, rf_prec, rf_rec, rf_cm = get_model_scores(rf_clf, X_train_prep, y_train, X_test_prep, y_test)

#F2 (beta=2): Mine safety, medical diagnosis, fraud detection
#F1 (beta=1): Balanced costs, general classification benchmarks
#F0.5 (beta=0.5): Spam filters, content moderation (false alarm very costly)

def get_f_scores(model_preds, y_test):
    model_f1 = f1_score(y_test, model_preds, zero_division=0)
    model_f2 = fbeta_score(y_test, model_preds, beta=2, zero_division=0)
    model_f05 = fbeta_score(y_test, model_preds, beta=0.5, zero_division=0)

    return model_f1, model_f2, model_f05


lr_f1, lr_f2, lr_f05 = get_f_scores(lr_preds, y_test)

rf_f1, rf_f2, rf_f05 = get_f_scores(rf_preds, y_test)


#For mine safety -> we should use F2 (beta=2)
#Missing a hazard costs orders of magnitude more than a false alarm
#beta=2 weights recall 4× more than precision')

#F1 hides the asymmetry — it would rank a model with P=0.9, R=0.1 (harmonic mean is symmetric by design)




#--------------------------------------------------------------------------------------------------------------

#ROC-AUC vs Average Precision

def get_predict_proba(model, X_test, y_test):
    y_proba_model = model.predict_proba(X_test)[:, 1]
    
    #ROC
    fpr_model, tpr_model, _ = roc_curve(y_test, y_proba_model)
    auc_model = roc_auc_score(y_test, y_proba_model)
    #PR
    pre_model, rec_model, _ = precision_recall_curve(y_test, y_proba_model)
    ap_model = average_precision_score(y_test, y_proba_model)

    return fpr_model,tpr_model, auc_model, pre_model, rec_model, ap_model, y_proba_model


fpr_lr, tpr_lr, auc_lr, pre_lr, rec_lr, ap_lr, y_proba_lr = get_predict_proba(lr_clf, X_test_prep, y_test)

fpr_rf, tpr_rf, auc_rf, pre_rf, rec_rf, ap_rf, y_proba_rf = get_predict_proba(rf_clf, X_test_prep, y_test)



#ROC-AUC:LR = auc_lr, RF=auc_rf -> both look good
#Avg Prec: LR = ap_lr, RF = ap_rf  -> both have significant room for improvement

#Rare positive class OR FP cost -> FN cost -> we should use PR curve and Average Precision
#Balanced classes OR both error types equally costly -> we should use ROC-AUC

#Why ROC misleads? FPR = FP / (FP + TN). With 2,414 TN, the denominator is always large. Even 200 false alarms only moves FPR by 200/2414 = 0.08.

#The PR curve skips TN entirely since it only asks about positive predictions.






#----------------------------------------------------------------------------------------------------------------------

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


#Tuning threshold the right way: from cv out-of-fold predictions
cv_probs = cross_val_predict(lr_clf, X_train_prep, y_train, cv=skf, method='predict_proba')[:,1]

pre_cv, rec_cv, thr_cv = precision_recall_curve(y_train, cv_probs)
f2_cv = (5*pre_cv[:-1]*rec_cv[:-1]) / (4*pre_cv[:-1]+rec_cv[:-1] + 1e-9)
best_idx = np.argmax(f2_cv)
best_thr_cv = thr_cv[best_idx]

#Tuning the threshold the wrong way: data leakage
pre_te, rec_te, thr_te = precision_recall_curve(y_test, y_proba_lr)
f2_te = (5 * pre_te[:-1] * rec_te[:-1]) / (4 * pre_te[:-1] + rec_te[:-1] + 1e-9)
best_thr_test = thr_te[np.argmax(f2_te)]


y_pred_default  = (y_proba_lr >= 0.5).astype(int)
y_pred_cv_thr   = (y_proba_lr >= best_thr_cv).astype(int)
y_pred_test_thr = (y_proba_lr >= best_thr_test).astype(int)


def get_thr_score(y_pred, y_test):
    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test, y_pred, zero_division=0)
    f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)

    return p, r, f2

p_default, r_default, f2_default = get_thr_score(y_pred_default, y_test)
p_cv_thr, r_cv_thr, f2_cv_thr = get_thr_score(y_pred_cv_thr, y_test)
p_test_thr, r_cv_thr, f2_cv_thr = get_thr_score(y_pred_test_thr, y_test)


#CV-tuned: threshold found using out-of-fold probabilities — honest estimate
#Test-tuned: threshold found by scanning test set directly — leakage
#The test-tuned threshold exploits test set knowledge and will not hold in production

#Why lowering the threshold increases recall?
#Lower threshold = model raises alarm at lower confidence
#More positives predicted -> more true positives caught (increases recall)
#But also more false positives (decreases precision)
#This is the fundamental precision-recall tradeoff





#--------------------------------------------------------------------------------------------------

base = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

# cross_val_score: one metric per fold
cv_scores = cross_val_score(base, X_train_prep, y_train, cv=skf, scoring='recall')

# cross_val_predict: one prediction per instance (out-of-fold)
cv_preds = cross_val_predict(base, X_train_prep, y_train, cv=skf)

#Wrong way: Predicting on training data
base.fit(X_train_prep, y_train)
train_preds = base.predict(X_train_prep)

rec_cv_pred = recall_score(y_train, cv_preds, zero_division=0)
rec_train = recall_score(y_train, train_preds, zero_division=0)


#cross_val_predict recall (honest): rec_cv_pred
#.predict(X_train) recall (inflated): rec_train
#Inflation from training data leak: rec_train - rec_cv_pred

#When to use each?
#cross_val_score:comparing models, reporting overall CV performance
#cross_val_predict: building confusion matrix, tuning thresholds, plotting PR/ROC
#on training data -> needs out-of-fold predictions
#.predict(X_train): NEVER for evaluation — model has already seen this data




#--------------------------------------------------------------------------------------------------

#Stratified KFold on Imbalanced Data


def compute_positive_rate(y, indices):
    return y.iloc[indices].mean()


def evaluate_fold_distributions(X, y, n_splits=5, threshold=0.04, random_state=42):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = []

    for fold_idx, ((_, val_idx_kf), (_, val_idx_skf)) in enumerate(
        zip(kf.split(X), skf.split(X, y)), start=1
    ):

        kf_rate = compute_positive_rate(y, val_idx_kf)
        skf_rate = compute_positive_rate(y, val_idx_skf)

        # Flag low distribution issue
        flag = "TOO LOW" if kf_rate < threshold else ""

        results.append({
            "fold": fold_idx,
            "kfold_rate": kf_rate,
            "stratified_rate": skf_rate,
            "flag": flag
        })

    return pd.DataFrame(results)



results_df = evaluate_fold_distributions(X_train_prep, y_train)
print(results_df)

regular_kf = KFold(n_splits=5, shuffle=True, random_state=42)
stratified_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


#Max deviation from target:
#Regular KFold: max(abs(r - y_train.mean()) for r in reg_rates)
#Stratified KFold: max(abs(r - y_train.mean()) for r in strat_rates)

#When a fold has near-zero positive rate
#Training fold: model sees almost no hazards during that fold
#Validation fold: recall computed on almost no positives — unreliable score

#scikit-learn uses StratifiedKFold by default for classifiers in cross_val_score.
#But if you manually create KFold() and pass it, you LOSE this protection.
#Always use StratifiedKFold explicitly for imbalanced classification



#-------------------------------------------------------------------------------------------------------


X_tr_num = X_train[NUMERIC].values
X_te_num = X_test[NUMERIC].values

sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr_num)
X_te_sc = sc.transform(X_te_num)

for model, needs_scale in [
    (sgd_clf, True),
    (lr_clf, True),
    (rf_clf, False)]:

    model.fit(X_tr_num, y_train)
    r_unsc = recall_score(y_test, model.predict(X_te_num), zero_division=0)

    model.fit(X_tr_sc, y_train)
    r_sc = recall_score(y_test, model.predict(X_te_sc), zero_division=0)

    delta = r_sc - r_unsc
    expected = '(high impact expected)' if needs_scale else '(no impact expected)'
    
    print(f'{r_unsc} : {r_sc} : {delta} : {expected}')


#Why gradient/distance models are sensitive to scale?
#energy ranges from 0 to 10^8, gpuls ranges 0 to ~700
#Gradient steps in energy dimensions are 10^5 times larger
#The loss surface is badly conditioned — convergence is slow or wrong

#Why tree models are scale-invariant?
#Trees split on thresholds: if energy > 5000
#Multiplying all energy values by 100 shifts the threshold, not the split logic
#The model learns the same decision boundary regardless of scale
#Models requiring scaling: SGD, LR, SVM, KNN, Neural Networks
#Models not requiring scaling: Decision Tree, Random Forest, Gradient Boosting, Naive Bayes




#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Final evaluation using optimal CV-tuned threshold


summary = []
for name, model in [('LogisticRegression', lr_clf), ('RandomForest', rf_clf) ]:
    # CV threshold tuning
    cv_pr = cross_val_predict(model, X_train_prep, y_train, cv=skf, method='predict_proba')[:, 1]
    p_cv, r_cv, t_cv = precision_recall_curve(y_train, cv_pr)
    f2s = (5*p_cv[:-1]*r_cv[:-1]) / (4*p_cv[:-1]+r_cv[:-1]+1e-9)
    opt_thr = t_cv[np.argmax(f2s)]

    # Final test
    model.fit(X_train_prep, y_train)
    y_prob = model.predict_proba(X_test_prep)[:, 1]
    y_pred = (y_prob >= opt_thr).astype(int)

    p  = precision_score(y_test, y_pred, zero_division=0)
    r  = recall_score(y_test, y_pred, zero_division=0)
    f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
    cm_final = confusion_matrix(y_test, y_pred)
    _, fp_, fn_, tp_ = cm_final.ravel()

    summary.append({'name':name, 'p': p, 'r': r, 'f2': f2, 'tp': tp_, 'fp': fp_, 'fn': fn_})

    print(f'{name} (threhold={opt_thr:.3f})')
    print(f'  Precision:       {p:.3f}  ({fp_} false alarms)')
    print(f'  Recall:          {r:.3f}  ({tp_}/{tp_+fn_} hazards caught)')
    print(f'  F2 (primary):    {f2:.3f}')
    print(f'  Missed hazards:  {fn_}')


best = max(summary, key=lambda x: x['f2'])

print(f'Best model: {best["name"]}')
print(f'Catches {best["tp"]}/{best["tp"]+best["fn"]} hazardous shifts (recall={best["r"]:.1%})')
print(f'Generates {best["fp"]} false alarms over the test period')
print(f'Misses {best["fn"]} hazards — each a potential rockburst with workers present')
