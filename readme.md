# ILogicEval construction
symdata folder contains predefined templates for transformation to natural language

Order of running the files:
1) make_expression.py
2) snli_preprocessing.py
3) mnli_preprocessing.py
4) dataset_formation_3c1e_3e1c.py
5) dataset_formation_missing.py
6) to_train_test_val.py (can create ILogicEval and s-ILogicEval)


# As a Pre-training Corpus in Other Logic Reasoning Task
Result can be found in EvalAI ReClor Leaderboard

Run the file to create f-ILogicEval (only need to remove instance in train.json): **f-ILogicEval.py**
1) git clone MERIt GitHub repo and download deberta-v2-xlarge in MERIt/pretrained-models/
```git clone https://github.com/SparkJiao/MERIt.git```
2) put the two files: deberta_mynew.yaml and deberta_mynew2.yaml in putToMERItConf folder to MERIt/conf/deberta_v2/
- extra pretraining (change the train_file, dev_file, test_file in deberta_mynew.yaml to run on f-ILogicEval and s-ILogicEval)
```python reclor_trainer_base_v2.py seed=4321 -cp conf/deberta_v2 -cn deberta_mynew.yaml```
- finetune on ReClor task
```python reclor_trainer_base_v2.py seed=4321 -cp conf/deberta_v2 -cn deberta_mynew2.yaml```


# Performance in Traditional LLMs
put the albert_mynew.yaml in putToMERItConf folder to MERIt/conf/albert/ and run

```python reclor_trainer_base_v2.py seed=4321 -cp conf/albert -cn albert_mynew.yaml```

put the mynew.yaml in putToMERItConf folder to MERIt/conf/roberta/ and run

```python reclor_trainer_base_v2.py seed=4321 -cp conf/roberta -cn mynew.yaml```


# Performance in Top Performing LLMs
Run file: **runtopllms.py**

copy the print result to corresponding files, group the lines if one return is printed in multiple lines

Evaluation and metric computation: **topllms_eval.py**
