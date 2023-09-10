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


mnli_tr = load_dataset("multi_nli", split="train")
mnli_valid_match = load_dataset("multi_nli", split="validation_matched")
mnli_valid_mismatch = load_dataset("multi_nli", split="validation_mismatched")

sorted_mnli_tr = mnli_tr.sort('label')
sorted_mnli_valid_match = mnli_valid_match.sort('label')
sorted_mnli_valid_mismatch = mnli_valid_mismatch.sort('label')
del mnli_tr, mnli_valid_match, mnli_valid_mismatch


def merge(listOfDicts):
  # assume same keys in all dict & item in type list
    res = {}
    c = 0
    for k in listOfDicts[0].keys():
        if(k=='premise'):
            res['premise']=[]
            res['hypothesis']=[]
            res['neg_premise']=[]
            res['neg_hypothesis']=[]
        elif(k=='hypothesis'): continue
        else:
            res[k]=[]
        for d in listOfDicts:
            tmp=d[k]
            if(k=='hypothesis'): continue
            if(k=='premise'): 
                newtmp, newh, newneg, newhneg = [], [], [], []
                for i in tqdm(range(len(d[k]))):
#                     if(negate(d[k][i], d['hypothesis'][i])!='filtered'):
#                         print(negate(d[k][i], d['hypothesis'][i]), d[k][i], d['hypothesis'][i])
                    negres = negate(d[k][i], d['hypothesis'][i])
                    if(negres!='filtered'): 
                        newtmp.append(negres[0])
                        newh.append(negres[1])
                        newneg.append(negres[2])
                        newhneg.append(negres[3])
                    else: 
                        newtmp.append('filtered')
                        newh.append('')
                        newneg.append('')
                        newhneg.append('')
                #tmp=list(tqdm(map(negate, tmp)))
                tmp=newtmp
                res['hypothesis']+=newh
                res['neg_premise']+=newneg
                res['neg_hypothesis']+=newhneg
            res[k]+=tmp
    df = pd.DataFrame(res)
    df = df[df['premise'] != 'filtered']
    return df.to_dict(orient='list')


entail_mnli = merge([sorted_mnli_tr[sorted_mnli_tr['label'].index(0):sorted_mnli_tr['label'].index(1)], 
                     sorted_mnli_valid_match[sorted_mnli_valid_match['label'].index(0):sorted_mnli_valid_match['label'].index(1)], 
                     sorted_mnli_valid_mismatch[sorted_mnli_valid_mismatch['label'].index(0):sorted_mnli_valid_mismatch['label'].index(1)]])
# entail_snli = merge([sorted_snli_tr[sorted_snli_tr['label'].index(0):sorted_snli_tr['label'].index(1)], sorted_snli_ts[sorted_snli_ts['label'].index(0):sorted_snli_ts['label'].index(1)], sorted_snli_valid[sorted_snli_valid['label'].index(0):sorted_snli_valid['label'].index(1)]])
with open('entail_mnli.json', 'w') as f:
    json.dump(entail_mnli, f)



neural_mnli = merge([sorted_mnli_tr[sorted_mnli_tr['label'].index(1):sorted_mnli_tr['label'].index(2)], sorted_mnli_valid_match[sorted_mnli_valid_match['label'].index(1):sorted_mnli_valid_match['label'].index(2)], sorted_mnli_valid_mismatch[sorted_mnli_valid_mismatch['label'].index(1):sorted_mnli_valid_mismatch['label'].index(2)]])
# neural_snli = merge([sorted_snli_tr[sorted_snli_tr['label'].index(1):sorted_snli_tr['label'].index(2)], sorted_snli_ts[sorted_snli_ts['label'].index(1):sorted_snli_ts['label'].index(2)], sorted_snli_valid[sorted_snli_valid['label'].index(1):sorted_snli_valid['label'].index(2)]])
with open('neural_mnli.json', 'w') as f:
    json.dump(neural_mnli, f)


contra_mnli = merge([sorted_mnli_tr[sorted_mnli_tr['label'].index(2):], sorted_mnli_valid_match[sorted_mnli_valid_match['label'].index(2):], sorted_mnli_valid_mismatch[sorted_mnli_valid_mismatch['label'].index(2):]])
# contra_snli = merge([sorted_snli_tr[sorted_snli_tr['label'].index(2):], sorted_snli_ts[sorted_snli_ts['label'].index(2):], sorted_snli_valid[sorted_snli_valid['label'].index(2):]])
with open('contra_mnli.json', 'w') as f:
    json.dump(contra_mnli, f)
