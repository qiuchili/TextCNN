# TextCNN
## Introduction
This is a pytorch implementation of [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

We use 3 different methods of embedding and test the model on SST-1 dataset.

rand-CNN:Word vectors are randomly initialized and then modified during training.

static-CNN:We use pre-trained word vectors from glove.840B.300d and the vectors are kept static during training.

non-static-CNN:Same as above but the vectors are fine-tuned during training.
## Requirements
Python3

torch

torchtext
## Results
|Model|SST-1|
|---|---|
|rand|40.1|
|static|40.0|
|non-static|41.3|
