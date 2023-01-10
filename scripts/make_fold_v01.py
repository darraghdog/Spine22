import os
import sys
import platform
import pandas as pd
from utils import set_pandas_display
from skmultilearn.model_selection import IterativeStratification
set_pandas_display()
import os
import random
import numpy as np

DEFAULT_RANDOM_SEED = 2021

def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seedBasic(seed=0)

trnfile = 'datamount/train.csv'
trndf = pd.read_csv(trnfile)

# Remove toronto data
utfile = 'datamount/RSNA_Kaggle_Experiment.csv'
utdf = pd.read_csv(utfile)
trndf = trndf.loc[trndf['StudyInstanceUID'].isin(utdf.StudyInstanceUID)].reset_index(drop = True)


k_fold = IterativeStratification(n_splits=5, order=2)
X = trndf[['StudyInstanceUID']]
y = trndf.filter(like='C').values
trndf['fold'] = -1

for t, (_, test) in enumerate(k_fold.split(X, y)):
    trndf.loc[test, 'fold'] = t

cols = ['patient_overall'] + trndf.filter(like='C').columns.tolist()
print(trndf.groupby(trndf.fold)[cols].agg('mean'))
print(trndf.shape)
trndf.to_csv('datamount/train_folded_v01.csv', index = False)

'''
      patient_overall    C1    C2    C3    C4    C5    C6    C7
fold
0               0.469 0.073 0.141 0.032 0.053 0.076 0.129 0.191
1               0.466 0.076 0.141 0.035 0.053 0.073 0.132 0.188
2               0.469 0.073 0.144 0.035 0.053 0.076 0.129 0.188
3               0.472 0.076 0.144 0.032 0.053 0.076 0.132 0.191
4               0.478 0.076 0.141 0.035 0.050 0.073 0.132 0.191
(1705, 10)

'''



