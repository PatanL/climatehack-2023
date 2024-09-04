#!/bin/bash
FILE=2_training_resnet_weather
jupyter nbconvert $FILE.ipynb --to python 
python $FILE.py| tee "$(date +"%Y_%m_%d_%I_%M_%p").log"
