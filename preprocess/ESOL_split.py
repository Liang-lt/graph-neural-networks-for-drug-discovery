# split the ESOL dataset into train, validation and test sets

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('../data/ESOL.csv')
data = data[['smiles', 'measured log solubility in mols per litre']]
data.columns = ['smiles', 'logS']

train, test = train_test_split(data, test_size=0.2, random_state=42)
train, valid = train_test_split(train, test_size=0.2, random_state=42)

# save the split datasets as csv.gz files
train.to_csv('../data/ESOL_train.csv.gz', index=False, compression='gzip')
valid.to_csv('../data/ESOL_valid.csv.gz', index=False, compression='gzip')
test.to_csv('../data/ESOL_test.csv.gz', index=False, compression='gzip')

