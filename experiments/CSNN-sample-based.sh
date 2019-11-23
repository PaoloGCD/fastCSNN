#!/bin/bash
mkdir results/result0
python code/simplifiedCSNN/training.py
python code/simplifiedCSNN/labeling.py
python code/simplifiedCSNN/testing.py

