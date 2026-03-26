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

X = df.drop('hazard', axis=1)
y = df['hazard']

CATEGORICAL = [col for col in X.columns if X[col].dtype == 'object']
NUMERICAL = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

# print('Categorical feature values: ')
# for col in CATEGORICAL:
#     print(f' {col}: {sorted(df[col].unique())}')

# print('Energy features - scale ranges: ')
# for col in ['genergy', 'energy', 'maxenergy']:
#     print(f' {col}: {df[col].min():.0f} to {df[col].max():,.0f}')
# print()

# print('Energy spans 10^4 to 10^8 - 10000x range')
# print('This will dominate gradient-based models without scaling')



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# print('Stratified Train/Test Split')
# print(f'Train: {len(X_train)} rows | positive rate: {y_train.mean()*100:.1f}%')
# print(f'Test: {len(X_test)} rows | positive rate: {y_test.mean()*100:.1f}%')
# print()

'''
Without stratify=y, random chanve could put all 170 hazardous shifts.
In training with none in test - making evaluation impossible.
'''


#BUILDING PREPROCESSING PIPELINE
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), NUMERICAL),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
], remainder='drop')

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)
print(f'\n Feature Matrix shape after preprocessing: {X_train_prep.shape}')



