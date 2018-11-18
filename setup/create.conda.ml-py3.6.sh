#!/bin/bash

# create ml environment in anaconda

conda create -n ml-py3.6 python=3.6
conda activate ml-py3.6

curl https://repo.anaconda.com/pkgs/main/osx-64/blas-1.0-mkl.tar.bz2 --output blas-1.0-mkl.tar.bz2
mv blas-1.0-mkl.tar.bz2 ~/anaconda2/pkgs/.
conda install --offline ~/anaconda2/pkgs/blas-1.0-openblas.tar.bz2

conda install numpy pandas scipy scikit-learn matplotlib
conda install juptyer

pip install https://github.com/vearutop/builds/releases/download/tensorflow-1.8.0-cp36-cp36m-macosx_10_13_x86_64.whl/tensorflow-1.8.0-cp36-cp36m-macosx_10_13_x86_64.whl
pip install keras
