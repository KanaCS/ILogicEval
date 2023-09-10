#!/usr/bin/env python
import pandas as pd
import json
import math
model=''

with open('MERIt/mydatav3/test.json','r') as f:
    ts = pd.DataFrame(json.load(f))

tsl = ts['question_label'].tolist()


with open('0-shot-'+model+'.txt','r') as f:
    a = f.readlines()


res = [i.split() for i in a]

# check which line error, total number of line != 6000, nothing shd be print if no error
for i in range(len(res)):
    if(res[i][0]!=str(int(i/4))): 
        print(i)
        break


gt = [i[1] for i in res]
pred = [i[2] for i in res]


# totalno = len(res)/4
# totalno


char = ['A','B','C','D','N']
accls = []
absaccls = []
ansls = {}
record = {'A':[], 'B':[], 'C':[], 'D':[], 'N':[], 'en':[]}
correct = 0
for idx in range(len(gt)):
    if(gt[idx]==pred[idx]): correct+=1
    if (idx%4==0):
        for a in char:
            ansls[a] = 0
    if(pred[idx]!='A' and pred[idx]!='B' and pred[idx]!='C' and pred[idx]!='D'):
        pred[idx]='N'
    ansls[char[(char.index(pred[idx])+(idx%4))%4]]+=1
    if(idx%4==3): 
        # calc entropy
        en = 0
        for k in ansls.keys():
            record[k].append(ansls[k])
            if(ansls[k]!=0):
                ansls[k] = ansls[k]/4
                en += ansls[k]*math.log(ansls[k])/math.log(4)
        en = 1+en
        record['en'].append(en)
        # calc % correctness
        accls.append((correct/4)*en)
        if(correct==4): absaccls.append(1)
        else: absaccls.append(0)
        # reset correct
        correct = 0
    #if(idx==10): break


# particalcir, cir
print('particalcir, cir')
print(sum(accls)/len(accls), sum(absaccls)/len(absaccls))


# accls
qrecord = {'3e1c':[], '3c1e':[], 'missing':[]}
for i in range(len(accls[:])):
    for k in qrecord.keys():
        if(k in ts['question_label'][i]):
            qrecord[k].append(accls[i])

for k in qrecord.keys():
    print(k,end=' ')
    print(sum(qrecord[k])/len(qrecord[k]))


# absaccls
qrecord = {'3e1c':[], '3c1e':[], 'missing':[]}
for i in range(len(absaccls)):
    for k in qrecord.keys():
        if(k in ts['question_label'][i]):
            qrecord[k].append(absaccls[i])

for k in qrecord.keys():
    print(k,end=' ')
    print(sum(qrecord[k])/len(qrecord[k]))


ngt, npred = [], []
for i in range(len(gt)):
    if(i%4==0):
        ngt.append(gt[i])
        npred.append(pred[i])


cnt = 0
resdc = {'3e1c':0, '3c1e':0, 'missing':0}
totaldc = {'3e1c':0, '3c1e':0, 'missing':0}
for i in range(len(ngt)):
    correct = False
    if(ngt[i]==npred[i]): correct = True
    for k in resdc.keys():
        if(k in tsl[int(i/4)] and 'True' in res[i][-1]): # True indicate the prediction is in A/B/C/D
            totaldc[k]+=1
            if(correct): 
                resdc[k]+=1
                cnt+=1


# acc
print('acc')
print(cnt/totalno, resdc, totaldc)


for k in resdc.keys():
    print(k, resdc[k]/totaldc[k])
