#!/bin/bash
# Epochs, batch size,and accumulate batch size (allow for bigger batch size)
epochs_float=7
batch_size_float=32
acc_batch_float=8
epochs_QAT=40
batch_size_QAT=32
acc_batch_QAT=4
# Create venv environment
# python3 -m venv env
# source env/bin/activate
# pip install -r requirements.txt
#Clear screen 
clear
# Only train float with epochs and batch size and accumulate batch size
python image_train.py false $epochs_float $batch_size_float $acc_batch_float
# Only train QAT with epochs and batch size and accumulate batch size
python image_train.py true $epochs_QAT $batch_size_QAT $acc_batch_QAT