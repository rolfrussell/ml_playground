#!/bin/bash

# create ml environment in anaconda

conda create -n ml-py3.6 python=3.6
conda activate ml-py3.6

conda install numpy pandas scipy scikit-learn matplotlib
conda install jupyter

pip install tensorflow
pip install keras
