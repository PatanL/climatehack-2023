#!/bin/bash
jupyter nbconvert 2_training.ipynb --to python 
python 2_training.py| tee "$(date +"%Y_%m_%d_%I_%M_%p").log"