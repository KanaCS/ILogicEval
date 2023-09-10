#!/usr/bin/env python
import pandas as pd
import json
import random


with open('newdatasetv3.json') as f:
    newdataset = pd.DataFrame(json.load(f))



for i,row in newdataset.iterrows():
    for odx, opt in enumerate(row['opts']):
        opt = opt.strip()
        if(opt[-1]!='.'): newdataset.loc[i,'opts'][odx] = opt+"."


#len(newdataset), len(newdataset[newdataset['question_label']=='3c1e']), len(newdataset[newdataset['question_label']=='3e1c'])



pd.set_option('display.max_colwidth', None)


newdataset['newpsg']=['' for i in range(len(newdataset))]


## ILogicEval
newdataset = newdataset.rename(columns={'psg':'context', 'opts':'answers', 'ans':'label'})
## s-ILogicEval
# newdataset = newdataset.rename(columns={'exp':'context', 'optstmts':'answers', 'ans':'label'})


#len(newdataset)*0.1

nrow = newdataset.shape[0]
train = newdataset[:nrow-3000]
val = newdataset[nrow-3000:nrow-1500]
test = newdataset[nrow-1500:nrow]


train['id_string'] = "train_"+train.index.astype(str)
val['id_string'] = "val_"+val.index.astype(str)
test['id_string'] = "test_"+test.index.astype(str)

train = train.sample(frac=1).reset_index()
val = val.sample(frac=1).reset_index()
test = test.sample(frac=1).reset_index()

train = train.drop('index',axis=1)
val = val.drop('index',axis=1)
test = test.drop('index',axis=1)


qe = [
    'Based on the information given, which is the most accurate conclusion?',
    'Which following statements is supported by the information presented above?',
    'According to the statements above, which is most likely to be true?',
    'If the above statements are accurate, which is the most reasonable inference?',
    'Which is the most plausible conclusion based on the information provided above?',
    'Based on the above statements, which can be inferred?',
    'Which is consistent with the information presented above?',
    'Which is the best interpretation of the statements above?',
    'According to the above statements, which is a valid conclusion?',
    'Which is most likely to be true based on the information given above?',
    'If the statements above are true, which shall also be true on the basis of them?',
    'If the above statements hold true, which  can be reasonably inferred?',
    'Which of the following logically follows from the information presented above?',
    'Based on the statements provided, which options is most consistent with them?',
    'Taking into account the information above, which is the most plausible conclusion?',
    'If we assume the above statements to be true, which is a valid inference?'
]
qc = [
    'Which statements below, if true, casts doubt on the argument?',
    'If the statements above are true, which shall be false on the basis of them?',
    'Based on the information given, which is the most inaccurate conclusion?',
    'Which of the following statements, if true, would weaken the argument presented?',
    'Given the information provided, which conclusions is least likely to be accurate?',
    'Assuming the statements above are accurate, which contradicts the argument?',
    #'If the given statements are assumed to be true, which conclusion would be least supported by the evidence?',
    'Which of the following, if true, would cast doubt on the validity of the argument?',
    'Considering the information provided, which statements is most likely to be false?',
    'Which of the following conclusions is least supported by the given information?',
    'If the above statements are accurate, which undermines the argument?',
    'Based on the given statements, which conclusions is least justified?',
    'Based on the statements provided, which options is least consistent with them?',
    'Based on the statements provided, which options is most inconsistent with them?',
    'Which logically contradicts to the information presented above?',
    'Taking into account the information above, which is the most implausible conclusion?',
    'If we assume the above statements to be true, which is a invalid inference?'
]


qmp=[
    'Which one of the following, if true, most strongly supports the conclusion above?',
    'Which one of the following would it be most relevant to investigate in evaluating the conclusion in above?',
    'Which one of the following is the missing premise leading to the conclusion?',
    'Which one of the following, if true, most strengthens the argument?',
    'What is the critical assumption necessary for the conclusion to be valid?',
    'Which one of the following, when considered valid, would support the conclusion the greatest?',
    'What is the underlying assumption necessary to bridge the gap between the statements and the conclusion above?',
    'What is the missing premise, needed to justify the conclusion?',
    'What is the missing assumption that connects the statements to the conclusion?',
    'What essential assumption is required to validate the conclusion?',
    'What underlying premise must be present to uphold the conclusion?',
    'Among the following, which one, if accurate, would most effectively strengthen the conclusion drawn above?',
    'Determine the omitted premise that serves as a basis for the conclusion provided.'
]



train['question'] = [random.choice(qe) if train.loc[i]['question_label']=='3c1e' 
                     else random.choice(qc) if train.loc[i]['question_label'] == '3e1c'
                     else random.choice(qmp) if 'missing_premise' in train.loc[i]['question_label']  
                     else None for i in range(len(train))]
val['question'] = [random.choice(qe) if val.loc[i]['question_label']=='3c1e' 
                   else random.choice(qc) if val.loc[i]['question_label'] == '3e1c'
                   else random.choice(qmp) if 'missing_premise' in val.loc[i]['question_label']  
                   else None for i in range(len(val))]
test['question'] = [random.choice(qe) if test.loc[i]['question_label']=='3c1e' 
                    else random.choice(qc) if test.loc[i]['question_label'] == '3e1c'
                    else random.choice(qmp) if 'missing_premise' in test.loc[i]['question_label']  
                    else None for i in range(len(test))]


train = train.reindex(columns=['context','question','answers','label','id_string'])
val = val.reindex(columns=['context','question','answers','label','id_string'])
test = test.reindex(columns=['context','question','answers','label','id_string'])



train = train[train['answers'].apply(lambda x: len(x)==4)]
val = val[val['answers'].apply(lambda x: len(x)==4)]
test = test[test['answers'].apply(lambda x: len(x)==4)]


get_ipython().system('mkdir MERIt/mydatav3')


train.to_json('MERIt/mydatav3/train.json', orient='records')
val.to_json('MERIt/mydatav3/val.json', orient='records')
test.to_json('MERIt/mydatav3/test.json', orient='records')