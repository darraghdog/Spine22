import sys
import os
import json
import platform
import glob
from tqdm import tqdm
import pandas as pd
PATH = '/mount/rsna2022'
'''
PATH = '/Users/dhanley/Documents/rsna2022'
os.chdir(f'{PATH}')
'''
from tqdm import tqdm 
from PIL import Image, ImageStat
from utils import PreprocessDataset
from torch.utils.data import Dataset, DataLoader
from utils import set_pandas_display
tqdm.pandas()
set_pandas_display()

mode="train"
data_dir = 'datamount'
ds = PreprocessDataset(mode="train", data_dir = 'datamount')
dl = DataLoader(ds, shuffle = False, batch_size=16, num_workers = 8)

for b in tqdm(dl, total = len(dl)):
    b


trndf = pd.read_csv(f'{data_dir}/train_folded_v01.csv')
dcmls = glob.glob(f'{data_dir}/{mode}*/**/*.dcm', recursive = True)
imgls = glob.glob(f'{data_dir}/{mode}*/**/*.jpg', recursive = True)

jsonls = glob.glob(f'{data_dir}/{mode}*/**/*.json', recursive = True)

dictit = lambda i: {**{'fname': i}, **json.loads(eval(open(i).read()))}
dd = [dictit(i) for i in tqdm(jsonls)]
dd = pd.DataFrame(dd)

dd['slice_num'] = dd.fname.str.split('/').str[-1].str.split('.').str[0].apply(int)
dd['z_pos'] = dd['2097202'].apply(eval).apply(lambda x: x[-1])



dd = dd.sort_values('524312 slice_num'.split())

dd['z_pos']


dd['patient_overall'] = trndf.set_index('StudyInstanceUID')\
                        .loc[dd['2097165'].apply(eval).values].patient_overall.values
dd['fold'] = trndf.set_index('StudyInstanceUID')\
                        .loc[dd['2097165'].apply(eval).values].fold.values

dd.iloc[0]
dd['2097165'].value_counts()

dd['z_dirn'] = dd.groupby('2097165')['z_pos'].transform(lambda x: (list(x)[-1]-list(x)[0])>0)

dd['z_pos'] - dd['z_pos'].shift(1)

ddd = dd.reset_index()['2097165 fold z_dirn patient_overall'.split()].drop_duplicates()
FOLD = 3
ct = pd.crosstab(ddd['z_dirn'], 
            ddd['patient_overall'])
ct

dd

dd

dd['2621444'].value_counts()
dd['2621488'].value_counts()


bbdf = pd.read_csv('datamount/train_bounding_boxes.csv')

dd



assert len(dcmls) == len(imgls)
assert len(dcmls) == len(jsonls)