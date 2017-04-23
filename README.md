# KBL-ANN
KBL-ANN is a deep learning script that predicting the match result in Korean baseball league by Artificial neural network with Tensorflow.

## Requirement

python3.6

tensorflow

pandas

numpy

## How to use?

```bash
pip3.6 install -r requirements.txt -I
cd KBL-ANN
python3.6 kbl_ANN.py
```

## How to review?

```bash
tensorboard --logdir=./logs/kbl_ANN_L0.0001_W1000_T10001
```

**Warning : the log folder name is depends on learning rate, weight size and traning times.**

## How did I make the data pre-processing?

please check out ```KBL-ANN/mining/kboDataPreprocessing.py```.
