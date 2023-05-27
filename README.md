# Implementation of NS-LCR
This is the official implementation of the paper "Learning Logic Rules as Explanations for Legal Case Retrieval" based on PyTorch.

## Overview

## NS-LCR(BERT)
## NS-LCR(BERT-PLI)
## NS-LCR(Lawformer)
## NS-LCR(BERT-ts-L1)

Parameters are set as default in the code.

### Get NS-LCR results in 3 steps: 
1. Get the results of the baseline model through training and testing. 

2. Get the results of the law-level module and case-level module. 

3. Use the fusion module to get the NS-LCR results.

## Reproduction
Check the following instructions for reproducing experiments.

### Dataset
The Dataset details is shown in dataset file

### Quick Start
#### 1. Download data and pre-trained model.

#### 2. Train and evaluate our model:

#### Step 1：
You can get the results of four baselines by training and testing the models.

Take BERT, for example:

```bash
python BERT/bert_train.py  
python BERT/bert_train.py --data_type elam
python BERT/bert_test.py
python BERT/bert_test.py --data_type elam
```

Other models reference their folders.

#### Step 2：
You can get the law-level and case-level results with the following instructions.

Download our pre-trained predicate evaluation module, see the law-level folder.

```bash
python law-level/bert_fol.py  
python law-level/bert_fol.py  --data_type elam
python case-level/bert_sent_emb.py
python case-level/bert_sent_emb.py --data_type elam
```

#### Step 3：
You can get the NS-LCR results with the following instructions in the fusion folder:

```bash
python bert_better.py --data_type Lecard --method bert
python bert_better.py --data_type Lecard --method bertpli
python bert_better.py --data_type Lecard --method shaobert
python bert_better.py --data_type Lecard --method lawformer
python bert_better.py --data_type ELAM --method bert
python bert_better.py --data_type ELAM --method bertpli
python bert_better.py --data_type ELAM --method shaobert
python bert_better.py --data_type ELAM --method lawformer
```

### Requirements
```
astor>=0.8.1
bert-seq2seq>=2.3.6
boto3>=1.26.97
botocore>=1.29.97
certifi>=2022.12.7
charset-normalizer>=3.0.1
click>=8.1.3
cloudpickle>=2.2.1
colorama>=0.4.6
contextlib2>=21.6.0
contourpy>=1.0.7
cycler>=0.11.0
et-xmlfile>=1.1.0
filelock>=3.9.0
fonttools>=4.39.2
gensim>=3.8.1
huggingface-hub>=0.12.0
idna>=3.4
importlib-metadata>=6.3.0
importlib-resources>=5.12.0
jieba>=0.42.1
jmespath>=1.0.1
joblib>=1.2.0
json-tricks>=3.16.1
kiwisolver>=1.4.4
matplotlib>=3.7.1
networkx>=3.0
nltk>=3.8.1
nni>=2.10
numpy>=1.24.1
openpyxl>=3.1.1
packaging>=23.0
pandas>=1.5.3
Pillow>=9.4.0
pip>=22.3.1
prettytable>=3.7.0
psutil>=5.9.4
pyparsing>=3.0.9
python-dateutil>=2.8.2
PythonWebHDFS>=0.2.3
pytorch-pretrained-bert>=0.6.2
pytz>=2022.7.1
PyYAML>=6.0
regex>=2022.10.31
requests>=2.28.2
responses>=0.23.1
rouge>=1.0.1
s3transfer>=0.6.0
schema>=0.7.5
scikit-learn>=1.2.1
scipy>=1.10.1
sentence-transformers>=2.2.2
sentencepiece>=0.1.97
setuptools>=65.6.3
simplejson>=3.19.1
six>=1.16.0
smart-open>=6.3.0
termcolor>=2.2.0
textrank4zh>=0.3
threadpoolctl>=3.1.0
tokenizers>=0.13.2
torch>=1.10.0+cu111
torchaudio>=0.10.0+rocm4.1
torchvision>=0.11.0+cu111
tqdm>=4.64.1
transformers>=4.27.1
typeguard>=3.0.2
types-PyYAML>=6.0.12.9
typing_extensions>=4.4.0
urllib3>=1.26.14
wcwidth>=0.2.6
websockets>=11.0.1
wheel>=0.37.1
zipp>=3.15.0
```

### Environments
We conducted the experiments based on the following environments:
* CUDA Version: 11.4
* torch version: 1.10.0
* OS: Ubuntu 18.04.5 LTS
* GPU: NVIDIA Geforce RTX 3090
* CPU: Intel(R) Xeon(R) Silver 4214 CPU @ 2.20GHz
