#!/bin/bash

# create ml environment in anaconda

conda create -n ml-py3.7 python=3.7
conda activate ml-py3.7

conda install numpy pandas scipy scikit-learn matplotlib
conda install jupyter

pip install tensorflow
error:  tensorflow not supported on python 3.7 yet
https://github.com/tensorflow/tensorflow/issues/17022

pip install keras
