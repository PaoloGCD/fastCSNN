#!/bin/bash
mkdir results/result1
python code/simplifiedCSNN/training.py 1 5 5
python code/simplifiedCSNN/labeling.py 1
python code/simplifiedCSNN/testing.py 1

