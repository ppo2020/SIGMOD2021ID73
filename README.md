# SelNet-KDD2020
souce code of KDD20 submission

will be uploaded by Feb 23, 2020.

# Datasets
**face**: https://drive.google.com/open?id=16aNBnAGg6BrrTVxOXsjGnvlr89vJYX5q

**fasttext**:https://drive.google.com/open?id=1FuWBcNxT_wO5cSCTJXJ6a6JjfD_vhfSb

**youtube**:

```
mkdir face face/real_data fasttext_cos fasttext_cos/real_data fasttext_eu fasttext_eu/real_data youtube youtube/real_data
```

Download the above datasets and put them into the directory "real_data", e.g., face_originalData.npy -> ./face/real_data/

# Structure
```
.
├── data
│   ├── face
│   ├── fasttext_cos
│   ├── fasttext_eu
│   └── youtube
├── model
│   ├── Dispatcher.py
│   ├── __pycache__
│   ├── selnet.py
│   └── selnetpart.py
├── proc
│   ├── covertree
│   ├── randompartition
│   ├── shell
│   └── train
└── run
    ├── CoverTree
    ├── RandomPartition
    └── one
```

* data: the dataset and all training information
* model: model file, e.g., SelNet
* proc: scripts for generating training data, cover tree and random partition strategies.
* run: scripts to train and inference
  - one: run SelNet without partition strategies
  - CoverTree: run SelNet with Cover Tree partition
  - RandomPartition: run SelNet with Random partition
        

# Run

Here use **face** dataset as example.

## 1. Generating training data

```
cd ./proc/shell

./train.sh face
```
The script train.sh generates training, validation and testing data for SelNet without partition.

This might take a long time to finish.

## 2. Partition strategies

There are two options: cover tree partition or random partition.

### 2.1 Cover tree (CT)

```
./train_ct.sh face
```

### 2.2 Random partition (RP)

```
./train_cp.sh face
```

All data are stored in the directory ./data/face/train/

For other datasets, just replace "face" with "fasttext_cos", "fasttext_eu" or "youtube"

## 3. Train and inference

### 3.1 Run SelNet without partition
```
cd ./run/one/

python train_face_d128_2M_smallSel_huber_log.py

python predict_face_d128_2M_smallSel_huber_log.py

```

### 3.2 Run SelNet with CT

```
cd ./run/CoverTree

./train_one_face_d128_2M_smallSel_huber_log.sh

python predict_one_face_d128_2M_smallSel_huber_log.py
```

### 3.3 Run SelNet with RP
```
cd ./run/RandomPartition

./train_one_face_d128_2M_smallSel_huber_log.sh

python predict_one_face_d128_2M_smallSel_huber_log.py
```
