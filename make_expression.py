#!/usr/bin/env python

import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.headless = True
wd = webdriver.Chrome('chromedriver',options=chrome_options)


# rule1: (¬(a∧b)→c)→(¬a→c)
# 
# rule2: ((a∨b∨...)→c)→(a→c)
# 
# rule3: (a→b)→(¬b→¬a)


t2qDict = {'~1':'∧',
           '~2':'∨',
           '~3':'¬',
           '~4':'↔',
           '~5':'→',
           '~6':'∀',
           '~7':'∃'}

q2tDict = {}
for k,i in t2qDict.items():
    q2tDict[i]=k

def wdfind(by, name):
    elem = wd.find_elements(by, name)
    res = []
    for e in elem:
        res.append(str(e.text))
    return res

def addbyrule(exp, expa, chosen, rule):
    if(rule==0):
        for c in chosen[1:-1]:
            expa += '∧' + c
        if(exp!=''): exp += ','
        exp += '(¬('+expa+')→'+chosen[-1]+')'
    elif(rule == 1):
        for c in chosen[1:-1]:
            expa += '∨' + c
        if(exp!=''): exp += ','
        exp += '(('+expa+')→'+chosen[-1]+')'
    else:
        if(exp!=''): exp += ','
        exp += '('+expa+'→'+chosen[1]+')'
    return exp


candidates = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
allexp, allpos, allneg = [], [], []
for no in tqdm(range(10000)):
    exp = ''
    candidates_count = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
    candidates_use = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
    for espn in range(random.randint(2,4)):
        # premise
        probi = (candidates_count.max()+1)-candidates_count
        for i in range(len(candidates_use)):
            if(candidates_use[i]>=2): probi[i]=0.1
            if(candidates_use[i]>=3): probi[i]=0
        chosen = np.random.choice(candidates, 3, p=probi/probi.sum(), replace=False)
        random.shuffle(chosen)
        rule = random.randint(0,3)
        expa = chosen[0]
        exp = addbyrule(exp, expa, chosen, rule)

        # find the chosen candidates
        if(rule==2):
            chosen = chosen[:2]
    
        # update candidates_count
        for c in chosen[:-1]:
            candidates_count[np.where(candidates==c)[0]]+=2
            candidates_use[np.where(candidates==c)[0]]+=1
        candidates_count[np.where(candidates==c)[0]]-=1
        candidates_use[np.where(candidates==chosen[-1])[0]]+=1
        
    # build possible conclusion under the chosen candidates
    poss_conc = []
    allchosen = candidates[np.where((candidates_use != 0) & (candidates_use <3))[0]]
    # new part
    toaddexp = exp.split(',')
    toaddsym = random.sample(list(allchosen), random.randint(0,int(len(allchosen)/1.5)))
    for sym in toaddsym:
        if(random.randint(0,2)%3==0): sym = '¬'+sym
        toaddexp.insert(random.randint(0, len(toaddexp)), sym)
    exp = ','.join(toaddexp)

    for a in allchosen:
        for b in allchosen:
            if(b!=a):
                poss_conc.append('(¬'+a+'→¬'+b+')')
                poss_conc.append('(¬'+a+'→'+b+')')
                poss_conc.append('('+a+'→¬'+b+')')
                poss_conc.append('('+a+'→'+b+')')

    allexp.append(exp)
    pos, neg = [], []
    for conc in poss_conc:
        query = exp+'|='+conc
        for k in q2tDict:
            query = query.replace(k, q2tDict[k])
        url='https://www.umsu.de/trees/#'+query
        wd.get(url)
        # print(url)
        if 'does not entail' in wd.find_element(By.ID, "statusmsg").text:
            res = wdfind(By.ID, "model")
            valid = False
        else:
            res = wdfind(By.CLASS_NAME, "treeNode")
            valid = True
        # print(valid, wd.find_element(By.ID, "statusmsg").text, res)
        if(valid): pos.append(conc)
        else: neg.append(conc)
    allpos.append(pos)
    allneg.append(neg)

  
dict = {'exp': allexp, 'pos': allpos, 'neg': allneg}       
csv = pd.DataFrame(dict) 
#df.to_csv('2023logicform.csv') 


crosspos, samepos = [], []
for idx,(exp,pos,neg) in csv.iterrows():
    # print(idx,exp,pos,neg)
    cris = list(map(lambda x: [i for i in list(x) if i in "ABCDEFGH"], exp.split('),(')))
    crossp, samep = [], []
    for p in pos:
        newp = [i for i in list(p) if i in "ABCDEFGH"]
        # print(p,newp,list(map(lambda cri: set(newp).issubset(set(cri)), cris)))
        if True in list(map(lambda cri: set(newp).issubset(set(cri)), cris)):
            samep.append(p.strip())
        else:
            crossp.append(p.strip())
    samepos.append(samep)
    crosspos.append(crossp)


csv['crosspos']=crosspos
csv['samepos']=samepos


csv.to_csv('2023logicform3.csv')