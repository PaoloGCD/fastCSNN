#!/bin/bash
mkdir results/result2
python code/peterDiehlCSNN/training-PeterDiehl.py 2
python code/peterDiehlCSNN/labeling-PeterDiehl.py 2
python code/peterDiehlCSNN/testing-PeterDiehl.py 2
