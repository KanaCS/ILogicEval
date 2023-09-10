#!/usr/bin/env python
import json
import pandas as pd
from random import sample
# get the list of instance with #logic sym used > 4, randomly pick a subset and refine it
with open('MERIt/mydatav3/train.json') as f:
    trainv3 = pd.DataFrame(json.load(f))

def getLogicSym(exp):
    return list(set(exp.replace('(','').replace(')','').replace('¬','').replace('→',',').replace('∨',',').replace('∧',',').split(',')))

toremove = []
for i,row in trainv3.iterrows():
    if(len(getLogicSym(row['exp']))>5): toremove.append(i)


trainv3.drop(index=sample(toremove,int(len(toremove)/3.5))).reset_index().to_json('MERIt/mydatav3/8455_train.json', orient='records')