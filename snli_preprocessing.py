#!/usr/bin/env python


from datasets import load_dataset
import pandas as pd 
import json
import re
from tqdm import tqdm
import stanza
import nltk
from nltk import pos_tag
import inflect
inflect = inflect.engine()
from nltk.stem.wordnet import WordNetLemmatizer
# nlp = stanza.Pipeline('en')



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
    return sent, hyp, negated, hyp_negated



snli_tr = load_dataset("snli", split="train")
snli_ts = load_dataset("snli", split="test")
snli_valid = load_dataset("snli", split="validation")




Rentail_snli, Rneutral_snli, Rcontra_snli = {'premise':[], 'hypothesis':[], 'neg_premise':[]}, {'premise':[], 'hypothesis':[]}, {'premise':[], 'hypothesis':[]}
groupOf3 = {'premise':[], 'entail':[], 'neural':[], 'contradict':[], 'neg_premise':[], 'neg_entail':[], 'neg_contradict':[]}
groupOfn3 = {'premise':[], 'entail':[], 'neural':[], 'contradict':[], 'neg_premise':[], 'neg_entail':[], 'neg_contradict':[]}
e, n ,c, cnt = [], [], [], 0
start = 0
for snli in [snli_ts.to_dict(), snli_valid.to_dict(),snli_tr.to_dict()]:
    p = snli['premise'][start]
    tostop = len(snli['label']) # for snli_ts.to_dict() & snli_valid.to_dict()
    for i in tqdm(range(tostop+1-start)):
        # snli got the same premise in different and neighbor instance, so want to group the instance w/ same premise tgt
        # when come to the end of loop or when premise not the same as previous trial
        if(start+i>=tostop or snli['premise'][start+i]!=p):
            if(cnt == 1): 
                if(e!=[]): 
                    res = negate(p,e[0])
                    if(res!='filtered'):
                        Rentail_snli['premise'] = res[0]
                        Rentail_snli['hypothesis'] = res[1]
                        Rentail_snli['neg_premise'] = res[2]
                elif(n!=[]):
                    Rneutral_snli['premise'] = p
                    Rneutral_snli['hypothesis'] = n
                elif(c!=[]):
                    Rcontra_snli['premise'] = p
                    Rcontra_snli['hypothesis'] = c
                # update values
                if(start+i<tostop):#len(snli['label'])):
                    p = snli['premise'][start+i]
                    e, n ,c, cnt = [], [], [], 0
                    continue
            elif(cnt == 3 and e!=[] and c!=[] and n!=[]): groupOfX = groupOf3
            else: groupOfX = groupOfn3
            res_e, res_n, res_c, res_ne, res_nc = [], [], [], [], []
            newp, negp = p, ''
            if(e!=[]):
                for ei in e:
                    res = negate(p,ei)
                    if(res!='filtered'):
                        newp, newh, negp, negh = res
                        res_e.append(newh)
                        res_ne.append(negh)
                    else: 
                        res_e.append('filtered')
                        res_ne.append('filtered')
            if(n!=[]):
                res_n = []
                for ni in n:
                    res = negate(p,ni)
                    if(res!='filtered'):
                        newp, newh, negp, negh = res
                        res_n.append(newh)
                    else: res_n.append('filtered')
            if(c!=[]):
                res_c = []
                for ci in c:
                    res = negate(p,ci)
                    if(res!='filtered'):
                        newp, newh, negp, negh = res
                        res_c.append(newh)
                        res_nc.append(negh)
                    else: 
                        res_c.append('filtered')
                        res_nc.append('filtered')
            if('filtered' not in res_e and 'filtered' not in res_n and 'filtered' not in res_c):
                #print(newp, res_e, res_n, res_c, negp)
                groupOfX['premise'].append(newp) 
                groupOfX['entail'].append(res_e) 
                groupOfX['neural'].append(res_n)
                groupOfX['contradict'].append(res_c)
                groupOfX['neg_premise'].append(negp)
                groupOfX['neg_entail'].append(res_ne)
                groupOfX['neg_contradict'].append(res_nc)
            # update values
            if(start+i<tostop):#len(snli['label'])):
                p = snli['premise'][start+i]
                e, n ,c, cnt = [], [], [], 0
        if(start+i<tostop):# len(snli['label'])):
            if(snli['label'][start+i] == 0): e.append(snli['hypothesis'][start+i])
            elif(snli['label'][start+i] == 1): n.append(snli['hypothesis'][start+i])
            elif(snli['label'][start+i] == 2): c.append(snli['hypothesis'][start+i])
        cnt+=1


with open('snli_groupOf3.json', 'w') as f:
    json.dump(groupOf3, f)

with open('snli_groupOfn3.json', 'w') as f:
    json.dump(groupOfn3, f)

with open('Rentail_snli.json', 'w') as f:
    json.dump(Rentail_snli, f)

with open('Rneutral_snli.json', 'w') as f:
    json.dump(Rneutral_snli, f)

with open('Rcontra_snli.json', 'w') as f:
    json.dump(Rcontra_snli, f)