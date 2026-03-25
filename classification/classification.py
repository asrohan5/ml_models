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
