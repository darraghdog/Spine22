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
trndf = trndf.loc[~trndf['StudyInstanceUID'].isin(utdf.StudyInstanceUID)].reset_index(drop = True)


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
0               0.473 0.072 0.141 0.037 0.054 0.082 0.139 0.193
1               0.493 0.072 0.141 0.037 0.052 0.079 0.136 0.196
2               0.469 0.072 0.141 0.035 0.052 0.079 0.136 0.196
3               0.460 0.074 0.141 0.037 0.054 0.079 0.139 0.193
4               0.485 0.072 0.141 0.035 0.054 0.082 0.136 0.196
'''



import os
import sys
import platform
import pandas as pd
from utils import set_pandas_display
sys.path.append("configs")
sys.path.append("models")
sys.path.append("data")
sys.path.append("postprocess")

# PATH = '/mount/Spine22'
trnfile = 'datamount/train.csv'
utfile = 'datamount/RSNA_Kaggle_Experiment.csv'
trndf = pd.read_csv(trnfile)
utdf = pd.read_csv(utfile)

trndf['fold'] = 0 # Validation data
trndf.loc[trndf['StudyInstanceUID'].isin(utdf.StudyInstanceUID), 'fold'] =  1


trndf.to_csv('datamount/train_folded_v01.csv', index = False)

print(trndf['fold'].value_counts())

'''
1    1705 (Training data)
0     314 (Validation data)
'''


