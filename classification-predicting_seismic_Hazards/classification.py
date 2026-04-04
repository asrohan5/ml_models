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

print('Categorical feature values:')
for col in CATEGORICAL:
    print(f'  {col}: {sorted(df[col].unique())}')


#Energy features — scale range
for col in ['genergy', 'energy', 'maxenergy']:
    print(f'  {col}: {df[col].min():.0f} to {df[col].max():,.0f}')

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
print(f'\nFeature matrix shape after preprocessing: {X_train_prep.shape}')


#------------------------------------------------------------------------------------

def get_model_scores(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    model_preds = model.predict(X_test)
    model_acc = accuracy_score(y_test, model_preds)
    model_prec = precision_score(y_test, model_preds)
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

