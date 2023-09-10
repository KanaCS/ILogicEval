#!/usr/bin/env python
import json
import pandas as pd
import stanza
import re
nlp = stanza.Pipeline('en')


with open('datasetv3.json') as f:
    dataset = pd.DataFrame(json.load(f))


edataset = dataset[dataset['question_label'].apply(lambda x: x=='3c1e')]


# pop out the important premise that can lead to non-entailment
res = []
for idx, row in tqdm(edataset.iterrows()):
    ls = row['exp'].replace("[","").replace("]","").replace("'","").split(',')
    conc = row['optstmts'][row['ans']]
    found = False
    # try all expression in ls to see whether it is the important premise lead to the conclusion
    for i in range(len(ls)):
        bepopped = ls.pop(i)
        query = ','.join(ls)+'|='+conc
        for k in q2tDict:
            query = query.replace(k, q2tDict[k])
        url='https://www.umsu.de/trees/#'+query
        wd.get(url)
        # print(url)
        ls.insert(i, bepopped)
        if 'does not entail' in wd.find_element(By.ID, "statusmsg").text:
            # success
            res.append(bepopped)
            found = True
            break
    if(found==False):
        res.append(None)


# prevent similar instance as multiple instance can come from the same "exp"
prev = None
for i in range(len(res)):
    if(res[i]==prev and res[i]!=None):
        prev = res[i]
        res[i]=None
    else: prev = res[i]


with open('datasetv3.json') as f:
    dataset = pd.DataFrame(json.load(f))
edataset = dataset[dataset['question_label'].apply(lambda x: x=='3c1e')]

# check whether the remaining options putting as premise are still not entail to the conclusion 
optsls = []
c = 0
for idx, row in tqdm(edataset.iterrows()):
    bepopped = res[c]
    c+=1
    if(bepopped!=None):
        optsls.append([])
        ls = row['exp'].replace("[","").replace("]","").replace("'","").split(',')
        ls.remove(bepopped)
        opts = row['optstmts']
        conc = opts.pop(row['ans'])
        allvalid = True
        #print('new', opts)
        for opt in opts:
            if(opt.strip()=='contradict'):
                while(True and optsls[-1]!=None):
                    #tmp = random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
                    tmp = random.choice(list(set(row['exp'].replace('(','').replace(')','').replace('¬','').replace('∨',',').replace('∧',',').replace('→',',').split(','))))
                    if(random.randint(0,1)): tmp = '¬'+tmp
                    query = ','.join(ls)+','+tmp+'|='+conc.strip()
                    for k in q2tDict:
                        query = query.replace(k, q2tDict[k])
                    url='https://www.umsu.de/trees/#'+query
                    wd.get(url)
                    # print(url)
                    if 'does not entail' in wd.find_element(By.ID, "statusmsg").text:
                        optsls[-1].append(tmp)
                        break
#                 if(optsls[-1]==None): print('None')
#                 else : print(optsls[-1])
                #optsls[-1]=None
                #break
            else:
                query = ','.join(ls)+','+opt.strip()+'|='+conc.strip()
                for k in q2tDict:
                    query = query.replace(k, q2tDict[k])
                url='https://www.umsu.de/trees/#'+query
                wd.get(url)
                # print(url)
                if 'does not entail' in wd.find_element(By.ID, "statusmsg").text:
                    optsls[-1].append(opt)
                else: 
                    optsls[-1]=None
                    break
#                 if(optsls[-1]==None): print('None')
#                 else: print(optsls[-1])
        #print(optsls[-1])
    else: optsls.append(None)


# combine optsls and res to generate new ans and optstmts and question_label
optstmts, ans = [], []
for a,b in zip(optsls, res):
    if(a==None or b==None or len(a)!=3):
        optstmts.append(None)
        ans.append(None)
    else:
        ans.append(random.randint(0,3))
        a.insert(ans[-1], b)
        optstmts.append(a)


def splitbydot(psg):
    psg = psg.replace('..','.')
    ls = psg.split('.')
    ls = list(filter(lambda x: x != '', ls))
    ls = list(filter(lambda x: x != ' ', ls))
    toconcat = []
    for i in range(len(ls)):
        if(ls[i].strip()[0].isalpha==False):
            return False
        if (ls[i].strip()[0].lower()==ls[i].strip()[0]):
            toconcat.append(i)
    if(0 in toconcat): toconcat.remove(0)
    for i in toconcat[::-1]:
        bepopped = ls.pop(i)
        ls[i-1] += '.'+bepopped
    return ls


import pickle 
with open('optstmts.pkl','wb') as f: 
    pickle.dump(optstmts, f)


with open('datasetv3.json') as f:
    dataset = pd.DataFrame(json.load(f))
edataset = dataset[dataset['question_label'].apply(lambda x: x=='3c1e')]

newdataset = dataset.loc[[]].copy()
c,c2=0,0
for idx, row in dataset.iterrows():
    if(row['question_label']=='3c1e'):
        spsg = splitbydot(row['psg'])
        sexp = row['exp'].replace("[","").replace("]","").replace("'","").split(',')
        if(optstmts[c]!=None and spsg!=False and len(spsg)==len(sexp)):
            newrow = dataset.loc[idx]
            ls = newrow['exp'].replace("[","").replace("]","").replace("'","").split(',')
            ls = [i.strip() for i in ls]
            idxtopop = ls.index(res[c])
            ls.pop(idxtopop)
            opts = row['opts']
            conctxt = row['opts'].pop(row['ans'])
            conc = row['optstmts'].pop(row['ans'])
            ansAddBack = optstmts[c].pop(ans[c])
            if('contradict' in row['optstmts']): 
                sym = optstmts[c][row['optstmts'].index('contradict')]
                if('¬' in sym):
                    tmp = row['sym2txt'][sym.replace('¬','')]
                    if('hypothesis' in tmp):
                        nres = tmp['neg_hypothesis']
                        if(nres==None): tmp = tmp['neg_premise']
                        else: tmp = nres
                    else:
                        if(tmp['neg_entail']==[]): tmp = tmp['neg_premise']
                        else:
                            nres = random.choice(tmp['neg_entail'])
                            if(nres==None): tmp = tmp['neg_premise']
                            else: tmp = nres
                else:
                    tmp = row['sym2txt'][sym]
                    if('hypothesis' in tmp):
                        tmp = tmp['hypothesis']
                    else:
                        tmp = random.choice(tmp['entail'])
                opts[row['optstmts'].index('contradict')] = tmp
            optstmts[c].insert(ans[c], ansAddBack)
            premise = spsg.pop(idxtopop)
            opts.insert(ans[c],premise)
            newrow['psg']='.'.join(spsg)+'.'+random.choice([' Thus, ',' Therefore, ',' In conclusion, '])+conctxt[0].lower()+conctxt[1:]
            newrow['opts']=opts
            newrow['exp']=','.join(ls)+'|='+conc
            newrow['ans']=ans[c]
            newrow['optstmts']=optstmts[c]
            newrow['question_label']='missing_premise_'+str(c2)
            c2+=1
            newdataset = pd.concat([newdataset, pd.DataFrame(newrow).transpose()])
        else: newdataset = pd.concat([newdataset, dataset.iloc[[idx]]])
        print(len(newdataset))
        c+=1
    else:
        newdataset = pd.concat([newdataset, dataset.iloc[[idx]]])
        print(len(newdataset))


# pd.set_option('display.max_colwidth', None)
# newdataset[newdataset['question_label'].apply(lambda x: 'missing_premise_' in x)]


newdataset.to_json('newdatasetv3.json')