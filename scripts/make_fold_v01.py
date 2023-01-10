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


