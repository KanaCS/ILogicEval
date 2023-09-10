#!/usr/bin/env python
import pandas as pd
import random
import json
import time
model='' 
with open('MERIt/mydatav3/test.json') as f:
  ts = pd.DataFrame(json.load(f))

def inputformation(df, i, label):
  input=''
  chr2str = ['A','B','C','D']
  row = df[df['question_label'].apply(lambda x: label in x)].iloc[i]
  input+=row['context']+'\n'+row['question']+'\n'
  for odx, opt in enumerate(row['answers']):
    input+=chr2str[odx]+'. '+opt
    input+='\n'
  input+='Answer: '+chr2str[row['label']]
  return input

fewshots = inputformation(tr, 0 , '3c1e') + '\n\n' + inputformation(tr, 0 , '3e1c') + '\n\n' + inputformation(tr, 0 , 'missing') + '\n\n'

# zero-shot
cnt, cnttotal, ls = 0,0, []
chr2str = ['A','B','C','D']
startc, beginc = 1, True
circular = [[0,1,2,3],[1,2,3,0],[2,3,0,1],[3,0,1,2]]
for i, row in ts[12:].iterrows():
  for cdx, cir in enumerate(circular):
    if(cdx<startc and beginc):
      continue
    input = "You need to answer in the form of 'Answer: <A/B/C/D>' "+'\n'
    #input += fewshots
    input+=row['context']+'\n'+row['question']+'\n'
    gt = row['answers'][row['label']]
    #print(gt)
    rowa = [row['answers'][x] for x in cir]
    rowl = rowa.index(gt)
    for odx, opt in enumerate(rowa):
      input+=chr2str[odx]+'. '+opt
      if(odx!=3): input+='\n'
    #print(input, chr2str[rowl])
    res = chat_gpt(input)
    print(i, chr2str[rowl], res[8], res, res[8] in chr2str)
    ls.append(res)
    if(res[8] in chr2str):
      cnttotal+=1
      if(chr2str[row['label']]==res[8]): cnt+=1
  beginc = False # not going to the if(cdx<startc) again

# few-shot
cnt, cnttotal, ls = 0,0, []
chr2str = ['A','B','C','D']
startc, beginc = 1, True
circular = [[0,1,2,3],[1,2,3,0],[2,3,0,1],[3,0,1,2]]
for i, row in ts[12:].iterrows():
  for cdx, cir in enumerate(circular):
    if(cdx<startc and beginc):
      continue
    input = "You need to answer in the form of 'Answer: <A/B/C/D>' "+'\n'
    input += fewshots
    input+=row['context']+'\n'+row['question']+'\n'
    gt = row['answers'][row['label']]
    #print(gt)
    rowa = [row['answers'][x] for x in cir]
    rowl = rowa.index(gt)
    for odx, opt in enumerate(rowa):
      input+=chr2str[odx]+'. '+opt
      if(odx!=3): input+='\n'
    #print(input, chr2str[rowl])
    res = chat_gpt(input)
    print(i, chr2str[rowl], res[8], res, res[8] in chr2str)
    ls.append(res)
    if(res[8] in chr2str):
      cnttotal+=1
      if(chr2str[row['label']]==res[8]): cnt+=1
  beginc = False # not going to the if(cdx<startc) again