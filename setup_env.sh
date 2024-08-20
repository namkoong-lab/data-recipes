#!/bin/bash
## The following commands install the necessary packages for the data-recipes environment on Linux
conda create -n data-recipes python==3.9.19
conda activate data-recipes

## Install requirements for opt_algos
pip install numpy==1.21.0
pip install dragonfly-opt==0.1.7
pip install smac==2.2.0
pip install numpy==1.21.0
pip install pandas==2.0.3
pip install matplotlib==3.7.5
pip install seaborn==0.13.2
pip install PyQt5==5.15.11

## Install requirements for data mixing experiments
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install datasets==2.20.0
