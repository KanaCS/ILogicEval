#!/usr/bin/env python

# construct missing premise in external noteboo
import json
import pandas as pd
from collections import Counter
import random
from datasets import load_dataset
import json
from tqdm import tqdm
import re
from tqdm import tqdm
import stanza
import time
# nlp = stanza.Pipeline('en')
import nltk
from nltk.tokenize import word_tokenize
import inflect
inflect = inflect.engine()



with open('snli_groupOfn3.json') as f:
    snlin3 = pd.DataFrame(json.load(f))
with open('snli_groupOf3.json') as f:
    snli3 = pd.DataFrame(json.load(f))
with open('entail_mnli.json') as f:
    emnli = pd.DataFrame(json.load(f))

def pos(sent):
    return nltk.pos_tag(sent.split())

pd.set_option('display.max_colwidth', None)


# cleansing
for i,row in tqdm(snlin3.iterrows()):
    for k in snlin3.keys():
        if(type(row[k])!=list): rowk = [row[k]]
        else: rowk = row[k]
        itmsres = []
        for itmidx, itm in enumerate(rowk):
            if(itm==None): continue
            res = pos(itm)
            words = [x[0] for x in res]
            tg = [x[1] for x in res]
            # check if exist wo
            if('wo' in words):
                # if prev word is DT tag, pop it out
                if(tg[words.index('wo')-1]=='DT'): 
                    tg.pop(words.index('wo'))
                    words.pop(words.index('wo'))
                # else, replace it with will
                elif('neg' in k): words[words.index('wo')]='will'
                # else print it out
                else: print(k, itm)
                        # check if exist ca
            if('ca' in words):
                # if prev word is DT tag, pop it out
                if(tg[words.index('ca')-1]=='DT'): 
                    tg.pop(words.index('ca'))
                    words.pop(words.index('ca'))
                # if not neg, replace it with will
                elif('neg' in k): words[words.index('ca')]='can'
                # if ca n't, combine
                elif(words[words.index('ca')+1]=="n't"):
                    words.pop(words.index('ca')+1)
                    words[words.index('ca')]="can't"
                # else print it out
                else: print(k, itm)
            # ensure the 1st word is captalized
            words[0] = words[0][0].upper()+words[0][1:]
            # ensure there is a . in the end
            if(words[-1][-1]!='.'): words[-1] = words[-1]+'.'
            if('VBG' in tg and tg.index('VBG')<=5):
                vbgidx = tg.index('VBG')
                if(tg[vbgidx-1]!='VBZ' and tg[vbgidx-1]!='VBP' and tg[vbgidx-1]!='RB' and tg[vbgidx-1]!='PRP' and tg[vbgidx-1]!='VBD' and tg[vbgidx-1]!='VB' and tg[vbgidx-1]!='RP' 
                   and words[vbgidx-1]!="don't" and words[vbgidx-1]!="doesn't" and words[vbgidx-1]!="didn't" and vbgidx!=0
                   and tg[vbgidx-1]!='DT' and tg[vbgidx-1]!='IN' and 'VBP' not in tg[:vbgidx] and 'VBZ' not in tg[:vbgidx] and 'VB' not in tg[:vbgidx] and 'VBD' not in tg[:vbgidx]):
                    # first check whether any 'is'/'are'/'was'/'were' in the other keys
                    # if no, then use infect to decide whether 'is'/'are'/'was'/'were'
                    toinsert = ''
                    end = False
                    for k2 in snlin3.keys():
                        rowk2 = [row[k2]] if type(row[k2])!=list else row[k2]
                        for itm2 in rowk2:
                            if(itm2==None): continue
                            if('is' in itm2.split()): 
                                toinsert='is'
                                end = True
                                break
                            elif('are' in itm2.split()): 
                                toinsert='are'
                                end = True
                                break
                            else: toinsert=''
                        if(end): break
                    if(toinsert==''):
                        if(inflect.singular_noun(' '.join(words[:vbgidx]))): toinsert='are'
                        else: toinsert='is'
                    words.insert(vbgidx, toinsert)
#                     if (',' not in itm): print(' '.join(words))
            if (',' in itm): itm=None
            itmsres.append(' '.join(words))
        if(k=='premise' or k=='neg_premise'): snlin3.loc[i, k] = itmsres[0]
        else: snlin3.loc[i, k] = itmsres


# cleansing on emnli
todrop = []
for key in ['premise','hypothesis','neg_premise','neg_hypothesis']:
    for i, row in tqdm(emnli.iterrows()):
        if(row[key]==None): continue
        sent = row[key].replace('"','').strip()
        for w2rp in ['afterwards,', 'soon,', 'as a concern,', 'and so,', 'sometimes,', 'so anyways,', 'but,', 'as a result,', 'soon after,', 'so,', 'sorry,', 'and thus,', 'as was usual,', 'but equally,', 'afterwards', 'but', 'soon', 'so', 'therefore', 'as', 'or', 'and','uh', 'and uh','because','because', 'while', 'just because', 'um-hum and', "at least", 'right', "yeah supposedly", "when"]: 
            if(w2rp in sent and sent.index(w2rp)==0):
                sent = sent.replace(w2rp, '', 1)
            w2rp = w2rp[0].upper()+w2rp[1:].strip()
            if(w2rp in sent and sent.index(w2rp)==0):
                sent = sent.replace(w2rp, '', 1).strip()
        if('uh' in sent): todrop.append(i)
        if('um.' in sent): todrop.append(i)
        if('um' in sent): todrop.append(i)
        if('oh' in sent): todrop.append(i)
        if('Okay' in sent): todrop.append(i)
        if('yeah' in sent): todrop.append(i)
        if("When" in sent): todrop.append(i)
        emnli.loc[i, key] = sent
        if('. ' in sent[:-1] or '."' in sent[:-1] or '?' in sent or ',' in sent):
            todrop.append(i)

emnli = emnli.drop(index=sorted(list(set(todrop))))

def count(ws, w):
    cnt, ls = 0,[]
    for idx,i in enumerate(ws):
        if(i==w):
            cnt+=1
            ls.append(idx)
    return cnt, ls



mkeys = ['premise','hypothesis','neg_premise','neg_hypothesis']
for i,row in tqdm(emnli.iterrows()):
    for k in mkeys:
        itm = row[k]
        if(itm==None): continue
        res = pos(itm)
        words = [x[0] for x in res]
        tg = [x[1] for x in res]
        # check if exist wo
        if('wo' in words):
            # if prev word is DT tag, pop it out
            if(tg[words.index('wo')-1]=='DT'): 
                tg.pop(words.index('wo'))
                words.pop(words.index('wo'))
            # else, replace it with will
            elif('neg' in k): words[words.index('wo')]='will'
            # else print it out
            else: print(k, itm)
         # check if exist ca
        if('ca' in words):
            # if prev word is DT tag, pop it out
            if(tg[words.index('ca')-1]=='DT'): 
                tg.pop(words.index('ca'))
                words.pop(words.index('ca'))
            # if not neg, replace it with will
            elif('neg' in k): words[words.index('ca')]='can'
            # if ca n't, combine
            elif(words[words.index('ca')+1]=="n't"):
                words.pop(words.index('ca')+1)
                words[words.index('ca')]="can't"
            # else print it out
            else: print(k, itm)
        # ensure the 1st word is captalized
        words[0] = words[0][0].upper()+words[0][1:]
        # ensure there is a . in the end
        if(words[-1][-1]!='.'): words[-1] = words[-1]+'.'
        emnli.loc[i, k] = ' '.join(words)

#len(snlin3), len(snli3), len(emnli)



sf2cf = {"we're": "we are", 
 "we've": "we have", 
 "we'll": "we will",
 "i'm": "i am", 
 "i've": "i have", 
 "i'll": "i will",
 "they're": "they are", 
 "they've": "they have", 
 "they'll": "they will",
 "you've": "you are", 
 "you're": "you have", 
 "you'll": "you will",
 "he's": "he is", 
 "he'd": "he had", 
 "he'll": "he will",
 "she's": "she is", 
 "she'd": "she had",
 "she'll": "she will"}


def getposition(ls, subword):
    for idx, i in enumerate(ls):
        if(subword in i):
            return idx
    return None



todrop=[]
search_words = [ 'we', 'they', 'he', 'she', 'you', 'I']
mkeys = ['premise','hypothesis','neg_premise','neg_hypothesis']
for i,row in tqdm(emnli.iterrows()):
    for k in mkeys:
        itm = row[k]
        if(itm==None): continue
        itm = itm.strip()
        if(itm[0]=="'"): itm=itm[1:]
        if(itm[:3]=="1) "): itm=itm[3:]
        # before: remove duplicate items
        itm = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', itm, flags=re.IGNORECASE)
        itm = itm.replace("Yes it's it's it's ", "It is ").replace("Yes it's not it's it's ", "It is not ").replace("It it it's i'm ", "I am ").replace("Er or ","").replace("What i i i'm ","I am ").replace("I i'm i could", "I could").replace("What i i don't i'm ", "I am not ").replace("I didn't i'm ", "I am ").replace("I did i'm ", "I am not ").replace("I've not i've ", "I have not ").replace("We're we're ", "We are ").replace("We're not we're ", "We are not ").replace("It's i'm ", "I am").replace("I've i've ", "I have ").replace("I i'm ", "I am ").replace("i i'm ", "I am ").replace("I i've ", "I have ").replace("i i've ", "I have ").replace("I i ",'I ').replace(" i i ",' I ').replace("[Hillary] ","Hillary ").replace("that that's ", "that's ").replace(".'.", ".").replace("Barr also ","").replace("Don't i'm ", "I am not ").replace("Um ","").replace(" was has just had "," had ").replace(" was not has just had "," had not").replace("Yet and then ","").replace("Oh yes ",'').replace("you're you don't ","you don't ").replace("you're you do ", "you do ").replace(").",'.').replace("...","").replace("Then ","").replace("2) ","").replace(".'.",".").replace("[I]n ","")
        s_itm = itm.split()
        if(":" in s_itm): 
            if(getposition(s_itm, ":")==1): re.sub(r'^[^:]*:', '', itm)
            else: todrop.append(i)
        for k2 in sf2cf.keys():
            while(k2 in itm): 
                itm = itm.replace(k2, sf2cf[k2])
                # after: remove duplicate items
                itm = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', itm, flags=re.IGNORECASE)
                res = pos(itm)
            uk2 = k2[0].upper()+k2[1:]
            while(uk2 in itm): 
                itm = itm.replace(uk2, sf2cf[k2][0].upper()+sf2cf[k2][1:])
                # after: remove duplicate items
                itm = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', itm, flags=re.IGNORECASE)
                res = pos(itm)
        # get pos tag
        itm = itm.replace(" i "," I ")
        res = pos(itm)
        #words = [x[0] for x in res]
        tg = [x[1] for x in res]
        # check if more than 1 PRP while 2nd PRP
        cnt, tgidxs = count(tg, 'PRP')
        if(cnt>=2 and (tg[tgidxs[1]-1]=="VBP" or tg[tgidxs[1]-1]=="PRP")): todrop.append(i)
        if(re.search(r"(?<![a-zA-Z])'[^\s][^']*[^'](?![a-zA-Z])", itm)!=None): todrop.append(i)
        if(len(s_itm)<2): todrop.append(i)
#         print(itm)
#         time.sleep(0.05)
        emnli.loc[i, k] = itm


pre = 'cleansed_'
snlin3.to_json(pre+'snli_groupOfn3.json', orient='records')
snli3.to_json(pre+'snli_groupOf3.json', orient='records')
emnli.to_json(pre+'entail_mnli.json', orient='records')


pre = 'cleansed_'
with open(pre+'snli_groupOfn3.json') as f:
    snlin3 = pd.DataFrame(json.load(f))
with open(pre+'snli_groupOf3.json') as f:
    snli3 = pd.DataFrame(json.load(f))
with open(pre+'entail_mnli.json') as f:
    emnli = pd.DataFrame(json.load(f))



def cleansingsnli(df):
    search_words = ['I' ,'You', 'We', 'They', 'He', 'She', 'It']
    toremove = []
    for i, row in tqdm(df.iterrows()):
        for key in df.keys():
            if(row[key]==None): continue
            rowkey = row[key]
            if(key=='premise' or key=='neg_premise'):
                for word in search_words:
                    if(word.lower() in rowkey.lower().split()):
                        toremove.append(i)
            else:
                newc = rowkey
                if(newc==None): continue
                for string in rowkey:
                    if(string==None): continue
                    for word in search_words:
                        if(word.lower() in string.lower().split()):
                            newc.remove(string)
                            break
                df.loc[i,key] = newc
    return df.drop(list(set(toremove)))

def cleansingmnli(df):
    search_words = [ 'we', 'they', 'he', 'she', 'you']
    toremove = []
    for i, row in tqdm(df.iterrows()):
        for word in search_words:
            if(word in row['premise'].lower().split()):
                if(word in row['hypothesis'].lower().split()):
                    wordC = word[0].upper()+word[1:]
                    if(word=='he'):
                        he=random.choice(mnames)
                        for key in ['premise', 'hypothesis','neg_premise','neg_hypothesis']:
                            if(row[key]==None): continue
                            tmp = row[key][:-1].split()
                            while('he' in tmp or 'He' in tmp):
                                if('he' in tmp): tmp[tmp.index('he')]=he
                                if('He' in tmp): tmp[tmp.index('He')]=he
                            df.loc[i, key] = ' '.join(tmp)+'.'
                    if(word=='she'):
                        she=random.choice(fnames)
                        for key in ['premise', 'hypothesis','neg_premise','neg_hypothesis']:
                            if(row[key]==None): continue
                            tmp = row[key][:-1].split()
                            while('she' in tmp or 'She' in tmp):
                                if('she' in tmp): tmp[tmp.index('she')]=she
                                if('She' in tmp): tmp[tmp.index('She')]=she
                            df.loc[i, key] = ' '.join(tmp)+'.'
                    if(word=='you'):
                        you=random.choice(fnames+mnames)
                        for key in ['premise', 'hypothesis','neg_premise','neg_hypothesis']:
                            if(row[key]==None): continue
                            tmp = row[key][:-1].split()
                            while('you' in tmp or 'She' in tmp):
                                if('you' in tmp): tmp[tmp.index('you')]=you
                                if('You' in tmp): tmp[tmp.index('You')]=you
                            df.loc[i, key] = ' '.join(tmp)+'.'
                    if(word=='we' or word=='they'):
                        they=random.sample(fnames+mnames, 2)
                        they = they[0]+' and '+they[1]
                        for key in ['premise', 'hypothesis','neg_premise','neg_hypothesis']:
                            if(row[key]==None): continue
                            tmp = row[key][:-1].split()
                            while('they' in tmp or 'They' in tmp or 'we' in tmp or 'We' in tmp):
                                if('they' in tmp): tmp[tmp.index('they')]=they
                                if('They' in tmp): tmp[tmp.index('They')]=they
                                if('we' in tmp): tmp[tmp.index('we')]=they
                                if('We' in tmp): tmp[tmp.index('We')]=they
                            df.loc[i, key] = ' '.join(tmp)+'.'
                else: toremove.append(i)
    return df.drop(list(set(toremove)))



fnames = ['Olivia','Emma','Charlotte','Amelia','Sophia','Isabella','Ava','Mia','Evelyn','Luna','Carla','Alice','Emily','Chloe','Mila','Ella','Lily','Hazel','Madison','Grace']
mnames = ['Liam','Noah','Oliver','James','Elijah','William','Henry','Lucas','Benjamin','Theodore','Felix','Atticus','Cassius','Oliver','Hugo','Joseph','David','Jacob','Luke','Gabriel']

]:


emnli = cleansingmnli(emnli).reset_index(drop=True)

snlin3 = cleansingsnli(snlin3).reset_index(drop=True)
snli3 = cleansingsnli(snli3).reset_index(drop=True)


snli3 = snli3[snli3['entail'].apply(lambda x: len(x)!=0)].reset_index(drop=True)
snli3 = snli3[snli3['entail'].apply(lambda x: len(x[0])>2)].reset_index(drop=True)
snlin3g2 = snlin3[snlin3['entail'].apply(lambda x: len(x) > 2)].reset_index(drop=True)
snlin3eq2 = snlin3[snlin3['entail'].apply(lambda x: len(x) == 2)].reset_index(drop=True)
snlin3l2 = pd.concat([snli3, snlin3[snlin3['entail'].apply(lambda x: len(x) < 2 and len(x)!=0)].reset_index(drop=True)], ignore_index=True)


csv=pd.read_csv('symdata/2023logicform3.csv')



for val in ['crosspos', 'neg']:
    csv[val] = csv[val].map(lambda x: x.replace("[","").replace("]","").replace("'","").split(','))



def exp2sym(exp):
    res = []
    targetsym = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for i in exp:
        if(i in targetsym):
            res.append(i)
    return res


with open('symdata/¬(a∧b)→c.txt', 'r') as f:
    rulena = [line.strip() for line in f.readlines()]
with open('symdata/(a∨b)→c.txt', 'r') as f:
    ruleor = [line.strip() for line in f.readlines()]
with open('symdata/→.txt', 'r') as f:
    rulei = [line.strip() for line in f.readlines()]


def negate(sent, hyp):
    sent, hyp = sent.strip(), hyp.strip()
    res = nlp(sent)
    words = [w.text for s in res.sentences for w in s.words]
    tags = [w.upos for s in res.sentences for w in s.words]
    hres = nlp(hyp)
    hwords = [w.text for s in hres.sentences for w in s.words]
    htags = [w.upos for s in hres.sentences for w in s.words]
    negated = ''
    # add subject
    if (("VERB"==tags[0] or "AUX"==tags[0]) and ("VERB"==htags[0] or "AUX"==htags[0])):
        sent = "we "+sent[0].lower()+sent[1:]
        hyp = "we "+hyp[0].lower()+hyp[1:]
    elif ("VERB"==tags[0] or "AUX"==tags[0]):
        if(htags[0]=='PRON' or htags[0]=='NOUN' or htags[0]=='PROPN'): 
            sent = hwords[0]+" "+sent[0].lower()+sent[1:]
        else: return 'filtered'
    elif("VERB"==htags[0] or "AUX"==htags[0]):
        if(tags[0]=='PRON' or tags[0]=='NOUN' or tags[0]=='PROPN'): 
            hyp = words[0]+" "+hyp[0].lower()+hyp[1:]
        else: return 'filtered'
    res = nlp(sent)
    words = [w.text for s in res.sentences for w in s.words]
    tags = [w.upos for s in res.sentences for w in s.words]
    # filter out sentence with "," and "?"
    if (";" in words or "(" in words or "," in words or "?" in words or "If" in words or "if" in words):
        return 'filtered'
    # filter out sentence with no verb
    if ("VERB" not in tags):
        return 'filtered'
    # filter out sentence with "INTJ" in the begining
    if ("INTJ"==tags[0]):
        return 'filtered'
    # filter out sentence if verb occurs more than twice
    if (" ".join(tags).replace("PART VERB", "to_v").split().count("VERB")>1):
        return 'filtered'
    # remove not / n't
    if ("not" in sent):
        negated = re.sub(r'\s+not\b', '', sent, count=1)
    elif ("n't" in sent):
        negated = re.sub(r"n't\b", '', sent, count=1)
    # for "is <verb>", add "not" after "is"
    elif ("AUX" in tags): # non-past tense
        # words.insert(tags.index("AUX") + 1, "not")
        splited = sent.split()
        for i in range(len(splited)):
            if (words[tags.index("AUX")] in splited[i]):
                splited.insert(i+1, "not")
                break
        negated = " ".join(splited)
    # add don't
    else: #any(vt in tags for vt in ["VBZ", "VBP", "VBD"]):
        #words.insert(tags.index("VERB"), "don't")
        splited = sent.split()
        for i in range(len(splited)):
            if (words[tags.index("VERB")] in splited[i]):
                tg = pos_tag([splited[i]])[0][-1]
                if(tg=='VBD' or tg=='VBN'): splited.insert(i, "didn't")
                elif(splited[i][-1]=='s'): splited.insert(i, "doesn't")
                else: splited.insert(i, "don't")
                splited[i+1] = WordNetLemmatizer().lemmatize(splited[i+1],'v')
                break
        negated = " ".join(splited) 
    # for hyp_neg
    if ("not" in hyp):
        hyp_negated = re.sub(r'\s+not\b', '', hyp, count=1)
    elif ("n't" in hyp):
        hyp_negated = re.sub(r"n't\b", '', hyp, count=1)
    # for "is <verb>", add "not" after "is"
    elif ("AUX" in htags): # non-past tense
        # words.insert(tags.index("AUX") + 1, "not")
        splited = hyp.split()
        for i in range(len(splited)):
            if (hwords[htags.index("AUX")] in splited[i]):
                splited.insert(i+1, "not")
                break
        hyp_negated = " ".join(splited)
    elif("VERB" not in htags): hyp_negated = None
    else:
        splited = hyp.split()
        for i in range(len(splited)):
            if (hwords[htags.index("VERB")] in splited[i]):
                tg = pos_tag([splited[i]])[0][-1]
                if(tg=='VBD' or tg=='VBN'): splited.insert(i, "didn't")
                elif(splited[i][-1]=='s'): splited.insert(i, "doesn't")
                else: splited.insert(i, "don't")
                splited[i+1] = WordNetLemmatizer().lemmatize(splited[i+1],'v')
                break
        hyp_negated = " ".join(splited) 
    return sent, hyp, negated#, hyp_negated



def sym2txtf(snlin3g2idx, snlin3eq2idx, snlin3l2idx, emnliidx, end):
    # assign text to each logic symbol
    sym2txt = {}
    for t in symset:
        g2tonextif, eq2tonextif = False, False
        if(t[1]>2):
            if(snlin3g2idx<snlin3g2max):
                sym2txt[t[0]]=snlin3g2.loc[snlin3g2idx]
                snlin3g2idx+=1
            else:
                g2tonextif = True
        if(t[1]==2 or g2tonextif):
            if(snlin3eq2idx<snlin3eq2max):
                sym2txt[t[0]]=snlin3eq2.loc[snlin3eq2idx]
                snlin3eq2idx+=1
            else:
                eq2tonextif = True
        if(t[1]==1 or eq2tonextif):
            snliormnli = random.randint(0,1)
            nli = snlin3l2 if snliormnli else emnli
            nlimax, nliidx = snlin3l2max if snliormnli else emnlimax, snlin3l2idx if snliormnli else emnliidx
            if (nliidx < nlimax):
                sym2txt[t[0]]=nli.loc[nliidx]
                if(snliormnli): snlin3l2idx+=1
                else: emnliidx+=1
            else:
                snliormnli = abs(snliormnli-1)
                nli = snlin3l2 if snliormnli else emnli
                nlimax, nliidx = snlin3l2max if snliormnli else emnlimax, snlin3l2idx if snliormnli else emnliidx
                if (nliidx < nlimax):
                    sym2txt[t[0]]=nli.loc[nliidx]
                    if(snliormnli): snlin3l2idx+=1
                    else: emnliidx+=1
                else:
                    print("no further instance can be create")
                    end = True
    return sym2txt, snlin3g2idx, snlin3eq2idx, snlin3l2idx, emnliidx, end



def constructpsg(expls):
    psg = ''
    for exp in expls:
        exp = exp.strip().replace("(","").replace(")","")
        syms = exp2sym(exp)
        if(len(syms)==3):
            if('¬' in exp): #rulena
                ruletp = random.choice(rulena)
                ruletp = replaceSymInTp(ruletp, sym2txt, syms)
            else: #ruleor
                ruletp = random.choice(ruleor)
                ruletp = replaceSymInTp(ruletp, sym2txt, syms)
        elif(len(syms)==2): #rulei
            ruletp = random.choice(rulei)
            ruletp = replaceSymInTp(ruletp, sym2txt, syms)
        elif(len(syms)==1):
            # random select from pre or hyp
            # check if need negate or not
            if('¬' in exp): 
                pm = "neg_premise"
                en = "neg_entail"
                hy = "neg_hypothesis"
            else:
                pm = "premise"
                en = "entail"
                hy = "hypothesis"
            # build ls of potential sentences to be select from pre and hyp
            ls = []
            ls.append(sym2txt[syms[0]][pm])
            if("entail" in sym2txt[syms[0]].keys()): ls = ls + sym2txt[syms[0]][en]
            else: ls.append(sym2txt[syms[0]][hy])
            # present None being selected
            if(None in ls): 
                ls = list(set(ls))
                ls.remove(None)
            txt = random.choice(ls)
            ruletp = txt
            if(ruletp[-1]!='.'): ruletp+='.'
            if(ruletp[:4]=='but '): ruletp=ruletp[4:]
            ruletp = ruletp[0].upper()+ruletp[1:]
        else:
            return False
        psg += ruletp.strip() if psg=='' else ' '+ruletp.strip()
    return psg



# can also process the neg for contradict in snli, maybe later
def replaceSymInTp(ruletp, sym2txt, syms, rptype='e', psidx=None):
    # 1st sym
    abssyms0 = syms[0].replace("¬","")
    abssyms1 = syms[1].replace("¬","")
    oriruletp = ruletp
    if('NOT <AAA>' in oriruletp and "¬" not in syms[0]):
        if(rptype=='c' and psidx==0): 
            if(sym2txt[abssyms0]['neg_contradict']==[]): return False
            res = random.choice(sym2txt[abssyms0]['neg_contradict'])
            if(res!=None): txt=res
            else: return False
        elif(rptype=='a'): 
            if('hypothesis' not in sym2txt[abssyms0].keys() and sym2txt[abssyms0]['neg_entail']==[]): return False
            res = sym2txt[abssyms0]['neg_hypothesis'] if 'hypothesis' in sym2txt[abssyms0].keys() else random.choice(sym2txt[abssyms0]['neg_entail'])
            if(res!=None): txt=res
            else: return False
        else:
            txt = sym2txt[abssyms0]['neg_premise']
        if(txt[:4]=='but '): txt=txt[4:]
        # if not in 1st position and the second char of txt is not capital letter (less like a named entity)
        if(ruletp.index('NOT <AAA>')!=0 and txt[1].lower()==txt[1]): txt = txt[0].lower()+txt[1:]
        else: txt = txt[0].upper()+txt[1:]
        if(txt[-1]=='.'): txt = txt[:-1]
        ruletp = ruletp.replace('NOT <AAA>', txt)
    elif('NOT <AAA>' not in oriruletp and "¬" in syms[0]):
        if(rptype=='c' and psidx==0):
            if(sym2txt[abssyms0]['neg_contradict']==[]): return False
            res = random.choice(sym2txt[abssyms0]['neg_contradict'])
            if(res!=None): txt=res
            else: return False
        elif(rptype=='a'): 
            if('hypothesis' not in sym2txt[abssyms0].keys() and sym2txt[abssyms0]['neg_entail']==[]): return False
            res = sym2txt[abssyms0]['neg_hypothesis'] if 'hypothesis' in sym2txt[abssyms0].keys() else random.choice(sym2txt[abssyms0]['neg_entail'])
            if(res!=None): txt=res
            else: return False
        else:
            txt = sym2txt[abssyms0]['neg_premise']
        if(txt[:4]=='but '): txt=txt[4:]
        if(ruletp.index('<AAA>')!=0 and txt[1].lower()==txt[1]): txt = txt[0].lower()+txt[1:]
        else: txt = txt[0].upper()+txt[1:]
        if(txt[-1]=='.'): txt = txt[:-1]
        ruletp = ruletp.replace('<AAA>', txt)
    else:
        # random select from pre or hyp
        ls = []
        if(rptype=='c' and psidx==0):
            ls = ls + sym2txt[abssyms0]['contradict']
        else:
            ls.append(sym2txt[abssyms0]['premise'])
            if("entail" in sym2txt[abssyms0].keys()): ls = ls + sym2txt[abssyms0]['entail']
            else: ls.append(sym2txt[abssyms0]['hypothesis'])
        if(rptype=='a'): ls.pop(0)
        txt = random.choice(ls)
        if(txt[:4]=='but '): txt=txt[4:]
        if(ruletp.index('<AAA>')!=0 and txt[1].lower()==txt[1]): txt = txt[0].lower()+txt[1:]
        else: txt = txt[0].upper()+txt[1:]
        if(txt[-1]=='.'): txt = txt[:-1]
        if('NOT <AAA>' in ruletp):
            ruletp = ruletp.replace('NOT <AAA>', txt)
        else:
            ruletp = ruletp.replace('<AAA>', txt)
    # 2nd sym
    if('NOT <BBB>' in oriruletp and "¬" not in syms[1]):
        if(rptype=='c' and psidx==1):
            if(sym2txt[abssyms1]['neg_contradict']==[]): return False
            res = random.choice(sym2txt[abssyms1]['neg_contradict'])
            if(res!=None): txt=res
            else: return False
        elif(rptype=='a'): 
            if('hypothesis' not in sym2txt[abssyms1].keys() and sym2txt[abssyms1]['neg_entail']==[]): return False
            res = sym2txt[abssyms1]['neg_hypothesis'] if 'hypothesis' in sym2txt[abssyms1].keys() else random.choice(sym2txt[abssyms1]['neg_entail'])
            if(res!=None): txt=res
            else: return False
        txt = sym2txt[abssyms1]['neg_premise']
        if(txt[:4]=='but '): txt=txt[4:]
        if(ruletp.index('NOT <BBB>')!=0 and txt[1].lower()==txt[1]): txt = txt[0].lower()+txt[1:]
        else: txt = txt[0].upper()+txt[1:]
        if(txt[-1]=='.'): txt = txt[:-1]
        ruletp = ruletp.replace('NOT <BBB>', txt)
    elif('NOT <BBB>' not in oriruletp and "¬" in syms[1]):
        if(rptype=='c' and psidx==1): 
            if(sym2txt[abssyms1]['neg_contradict']==[]): return False
            res = random.choice(sym2txt[abssyms1]['neg_contradict'])
            if(res!=None): txt=res
            else: return False
        elif(rptype=='a'): 
            if('hypothesis' not in sym2txt[abssyms1].keys() and sym2txt[abssyms1]['neg_entail']==[]): return False
            res = sym2txt[abssyms1]['neg_hypothesis'] if 'hypothesis' in sym2txt[abssyms1].keys() else random.choice(sym2txt[abssyms1]['neg_entail'])
            if(res!=None): txt=res
            else: return False
        txt = sym2txt[abssyms1]['neg_premise']
        if(txt[:4]=='but '): txt=txt[4:]
        if(ruletp.index('<BBB>')!=0 and txt[1].lower()==txt[1]): txt = txt[0].lower()+txt[1:]
        else: txt = txt[0].upper()+txt[1:]
        if(txt[-1]=='.'): txt = txt[:-1]
        ruletp = ruletp.replace('<BBB>', txt)
    else:
        # random select from pre or hyp
        ls = []
        if(rptype=='c' and psidx==1):
            ls = ls + sym2txt[abssyms1]['contradict']
        else:
            ls.append(sym2txt[abssyms1]['premise'])
            if("entail" in sym2txt[abssyms1].keys()): ls = ls + sym2txt[abssyms1]['entail']
            else: ls.append(sym2txt[abssyms1]['hypothesis'])
        if(rptype=='a'): ls.pop(0)
        txt = random.choice(ls)
        if(txt[:4]=='but '): txt=txt[4:]
        if(ruletp.index('<BBB>')!=0 and txt[1].lower()==txt[1]): txt = txt[0].lower()+txt[1:]
        else: txt = txt[0].upper()+txt[1:]
        if(txt[-1]=='.'): txt = txt[:-1]
        if('NOT <BBB>' in ruletp):
            ruletp = ruletp.replace('NOT <BBB>', txt)
        else:
            ruletp = ruletp.replace('<BBB>', txt)
    # 3rd sym
    if(len(syms)>2):
        # random select from pre or hyp
        ls = []
        ls.append(sym2txt[syms[2]]['premise'])
        if("entail" in sym2txt[syms[2]].keys()): ls = ls + sym2txt[syms[2]]['entail']
        else: ls.append(sym2txt[syms[2]]['hypothesis'])
        txt = random.choice(ls)
        if(txt[:4]=='but '): txt=txt[4:]
        if(ruletp.index('<CCC>')!=0 and txt[1].lower()==txt[1]): txt = txt[0].lower()+txt[1:]
        else: txt = txt[0].upper()+txt[1:]
        if(txt[-1]=='.'): txt = txt[:-1]
        ruletp = ruletp.replace('<CCC>', txt)
    return ruletp



snlin3g2max = len(snlin3g2)
snlin3eq2max = len(snlin3eq2)
snlin3l2max = len(snlin3l2)
#snli3max = len(snli3)
emnlimax = len(emnli)
snlin3g2idx, snlin3eq2idx, snlin3l2idx, emnliidx = 0, 0, 0, 0 # snli3idx combined to snlin3l2idx
presnlin3g2idx, presnlin3eq2idx, presnlin3l2idx, preemnliidx = 0, 0, 0, 0
end = False
dataset = {'psg':[], 'opts':[], 'ans':[], 'exp':[], 'optstmts':[], 'question_label':[], 'sym2txt':[]}
for i, row in csv.iterrows():
    if(len(row['exp'].replace('(','').replace(')','').replace('¬','').replace('→',',').replace('∨',',').replace('∧',',').split(','))>=19): continue
    print(len(dataset['psg']),row['exp'])
    # exp
    symset = Counter(exp2sym(row['exp'])).most_common()
    symls = [x[0] for x in symset]
    
    # regenerate psg w.r.t each crosspos instance
    skip,skipc=False,0
    rowscp = row['crosspos']
    for posdx, pos in enumerate(rowscp):
        if(len(row['neg'])==0): break
        if(skip):
            skipc-=1
            if(skipc==0): skip=False
            continue
        eorc = random.randint(0,1)
        toe = False
        if(eorc): # which stmt contradicts (3 entail, 1 contradict)
            if(len(rowscp)<3 or len(rowscp)<=posdx+2): 
                toe=True
            else:
                skip=True
                skipc=2
                psymsls = []
                for posi in [rowscp[posdx], rowscp[posdx+1], rowscp[posdx+2]]:
                    psymsi = posi.strip().replace("(","").replace(")","").split('→')
                    psymsls.append(psymsi)
            if(not toe):
                # function sym2txtf to assign text from mnli dataset to sym
                presnlin3g2idx, presnlin3eq2idx, presnlin3l2idx, preemnliidx = snlin3g2idx, snlin3eq2idx, snlin3l2idx, emnliidx
                sym2txt, snlin3g2idx, snlin3eq2idx, snlin3l2idx, emnliidx, end = sym2txtf(snlin3g2idx, snlin3eq2idx, snlin3l2idx, emnliidx, end)
                if(end): break
                # create passage content
                psg = constructpsg(row['exp'].split(","))
                opts, optstmts = [], [rowscp[posdx], rowscp[posdx+1], rowscp[posdx+2]]
                for psyms in psymsls:
                    ansruletp = random.choice(rulei)
                    ansruletp = replaceSymInTp(ansruletp, sym2txt, psyms, 'a')
                    if(ansruletp==False):
                        # reset the assigned mnli txt
                        snlin3g2idx, snlin3eq2idx, snlin3l2idx, emnliidx = presnlin3g2idx, presnlin3eq2idx, presnlin3l2idx, preemnliidx 
                        toe = True
                        skip,skipc=False,0
                        break
                    opts.append(ansruletp)
                if(not toe):
                    neg = random.choice(row['neg'])
                    nsyms = neg.strip().replace("(","").replace(")","").split('→')
                    nruletp = random.choice(rulei)
                    nruletp = replaceSymInTp(nruletp, sym2txt, nsyms)
                    ans = random.randint(0,3)
                    opts.insert(ans, nruletp)
                    optstmts.insert(ans, neg)
                    if(len(opts)!=4):
                        # reset the assigned mnli txt
                        snlin3g2idx, snlin3eq2idx, snlin3l2idx, emnliidx = presnlin3g2idx, presnlin3eq2idx, presnlin3l2idx, preemnliidx 
                        continue
                    dataset['psg'].append(psg)
                    dataset['opts'].append(opts)
                    dataset['ans'].append(ans)
                    dataset['exp'].append(row['exp'])
                    dataset['optstmts'].append(optstmts)
                    dataset['question_label'].append('3e1c')
                    dataset['sym2txt'].append(sym2txt)
        elif(abs(eorc-1) or toe): # what stmt entails (3 contradict, 1 entail)
#             pass
            psyms = pos.strip().replace("(","").replace(")","").split('→')
            tonext = False
            for psym in psyms:
                if(psym.replace("¬","") not in symls): tonext = True
            if(pos=='' or tonext): continue
            # function sym2txtf to assign text from mnli dataset to sym
            presnlin3g2idx, presnlin3eq2idx, presnlin3l2idx, preemnliidx = snlin3g2idx, snlin3eq2idx, snlin3l2idx, emnliidx
            sym2txt, snlin3g2idx, snlin3eq2idx, snlin3l2idx, emnliidx, end = sym2txtf(snlin3g2idx, snlin3eq2idx, snlin3l2idx, emnliidx, end)
            if(end): break
            # create passage content
            psg = constructpsg(row['exp'].split(","))

            # ans
            #psyms = pos.strip().replace("(","").replace(")","").split('→')
            ansruletp = random.choice(rulei)
            ansruletp = replaceSymInTp(ansruletp, sym2txt, psyms, 'a')
            if(ansruletp==False): 
                # reset the assigned mnli txt
                snlin3g2idx, snlin3eq2idx, snlin3l2idx, emnliidx = presnlin3g2idx, presnlin3eq2idx, presnlin3l2idx, preemnliidx 
                continue

            # wrong options
            opts, optstmts = [], []
            # check if contradict exist
            psidx = random.randint(0,1)
            cruletp = False
            if('contradict' in sym2txt[psyms[psidx].replace('¬','')] and len(sym2txt[psyms[psidx].replace('¬','')]['contradict'])!=0):
                cruletp = random.choice(rulei)
                cruletp = replaceSymInTp(cruletp, sym2txt, psyms, 'c', psidx)
            elif('contradict' in sym2txt[psyms[abs(psidx-1)].replace('¬','')] and len(sym2txt[psyms[abs(psidx-1)].replace('¬','')]['contradict'])!=0):
                cruletp = random.choice(rulei)
                cruletp = replaceSymInTp(cruletp, sym2txt, psyms, 'c', abs(psidx-1))
            # add one irrelevant (cancelled)
            # the randomly draw from row['neg']
            if(cruletp==False): n=3
            else: n=2
            lenn = len(row['neg'])
            if(lenn<3):
                # reset the assigned mnli txt
                snlin3g2idx, snlin3eq2idx, snlin3l2idx, emnliidx = presnlin3g2idx, presnlin3eq2idx, presnlin3l2idx, preemnliidx 
                continue
            negs = random.sample(row['neg'], min(lenn, n+10))
            ndx=0
            for neg in negs:
                if(ndx==n): break
                tonext = False
                nsyms = neg.strip().replace("(","").replace(")","").split('→')
                for nsym in nsyms:
                    if(nsym.replace("¬","") not in symls): tonext = True
                if(tonext): continue
                nruletp = random.choice(rulei)
                nruletp = replaceSymInTp(nruletp, sym2txt, nsyms)
                opts.append(nruletp)
                optstmts.append(neg)
                ndx+=1
            if(cruletp!=False): 
                pdx = random.randint(0,n)
                opts.insert(pdx, cruletp)
                optstmts.insert(pdx, 'contradict')
            # insert correct ans and get ans label
            ans = random.randint(0,3)
            opts.insert(ans, ansruletp)
            optstmts.insert(ans, pos)
            if(len(opts)!=4):
                # reset the assigned mnli txt
                snlin3g2idx, snlin3eq2idx, snlin3l2idx, emnliidx = presnlin3g2idx, presnlin3eq2idx, presnlin3l2idx, preemnliidx 
                continue
            
            dataset['psg'].append(psg)
            dataset['opts'].append(opts)
            dataset['ans'].append(ans)
            dataset['exp'].append(row['exp'])
            dataset['optstmts'].append(optstmts)
            dataset['question_label'].append('3c1e')
            dataset['sym2txt'].append(sym2txt)


dataset = pd.DataFrame(dataset)

dataset.to_json('datasetv3.json')