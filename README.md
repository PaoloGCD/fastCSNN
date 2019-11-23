# Fast Convergence of Competitive Spiking Neural Networks with Sample-based Weight Initialization

This repository contains code implementation for the paper "Fast Convergence of Competitive Spiking Neural Networks with Sample-based Weight Initialization". The code uses sample-based weight initilialization to reduce the convergence time of Competitive Spiking Neural Networks (CSNN). This initialization method is tested in a CSNN with 784 sensory layer Poisson units, 400 Leaky Integrate and Fire neurons with conductance-based stimulation input, trace-based STDP, direct inhibition, weight normalization, and no resting period between each sample presentation. The convergence time and testing accuracy result is compare with the state of the art CSNN implemented by Peter Diehl in the paper "[Unsupervised learning of digit recognition using spike-timing-dependent plasticity](https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full#)".

## Installation

### Prerequisites

The code runs in Python2.7 using the neural network simulator package Brian. Specifically, the following packages are needed:

- Brian 1.4
- Numpy
- Opencv
- Tensorflow
- Matplotlib
- Weave
- Pickle

### Clone

Clone this repo to your local machine using `https://github.com/PaoloGCD/fastCSNN.git`

### Dataset

The experiments are performed on the MNIST dataset. From the repository directory, download and extract the dataset with:

```shell
$ cd data
$ wget -i MNIST.txt
$ gunzip *.gz
$ python ../code/misc/prepare-dataset.py
```

## Running the tests

A CSNN experiment runs in three sequential steps: training, labelig and testing. Each step is implemented into an independent file under the directory ./code/simplifiedCSNN for running the simplified CSNN or ./code/peterDiehlCSNN for running Peter Diehl's CSNN implementation. The simplified CSNN runs in total in approximately 10 hours all three steps which is half the time of Peter Diehl's implementation.

To run the simplified CSNN with sample-based initialization, from the root directory execute:

```shell
$ sh ./experiments/CSNN-sample-based.sh
```

A simplified CSNN with sample-based initialization using only class 5 initial weights is run using:

```shell
$ sh ./experiments/CSNN-sample-based-class-5.sh
```

Peter Diehl's CSNN implementation is run by:

```shell
$ sh ./experiments/CSNN-Peter-Diehl.sh
```

## Seeing results

After training, the training accuracy is obtained by running, from the root directory:

```shell
$ python ./code/evaluation/training-accuracy.py 0
```

After labeling and testing is finished, the test accuracy for maximum and confidence-based voting classifier methods are obtained with:

```shell
$ python ./code/evaluation/testing-accuracy.py 0 1
```

Finally, the test accuracy for an add-on neural network classifier is obtained by running:

```shell
$ python ./code/evaluation/testing-accuracy-nn.py 0
```

## Authors

* **Paolo G. Cachi** - *Virginia Commonwealth University* - USA

## Acknowledgments

* The code is based on Peter Diehl's [CSNN implementation](https://github.com/peter-u-diehl/stdp-mnist).
