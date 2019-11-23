import sys
import numpy as np


experiment_number = 0
number_experiments = 1
experiment_path = './results/'

if len(sys.argv) == 3:
    experiment_number = int(sys.argv[1])
    number_experiments = int(sys.argv[2])


# LOAD TRAINING LABELS
training_data_path = "./data/"
training_y = np.load(training_data_path + 'MNIST-training-labels.npy')


# LOAD TESTING LABELS
testing_data_path = "./data/"
testing_y = np.load(testing_data_path + 'MNIST-testing-labels.npy')


# GET RUNNING ACCURACY
accuracy = np.zeros((number_experiments, 2))
for index, experiment in enumerate(range(experiment_number, experiment_number+number_experiments)):

    # LOAD RATES
    labeling_rates = np.load(experiment_path + 'result' + str(experiment) + '/labeling_rates.npy')
    testing_rates = np.load(experiment_path + 'result' + str(experiment) + '/testing_rates.npy')

    labeling_rates = labeling_rates / labeling_rates.sum(axis=1).reshape(60000, 1)
    testing_rates = testing_rates / testing_rates.sum(axis=1).reshape(10000, 1)

    # GET NEURON LABELS
    number_classes = 10
    number_samples = labeling_rates.shape[0]
    excitatory_neurons = labeling_rates.shape[1]
    total_class_activity = np.zeros((number_classes, excitatory_neurons))

    # get neuron activity per class
    for i in xrange(number_classes):
        class_spike_activity = labeling_rates[training_y[:number_samples].reshape(-1) == i, :]
        total_class_activity[i, :] = np.sum(class_spike_activity, axis=0)/float(class_spike_activity.shape[0])

    # max voting
    class_rank = np.argsort(total_class_activity, axis=0)
    labels_max_voting = class_rank[-1, :]

    # confidence voting
    labels_confidence_voting = np.copy(total_class_activity)

    # PREDICT LABELS MAX VOTING
    number_samples = testing_rates.shape[0]
    total_class_activity = np.zeros((number_samples, number_classes))

    # get neuron activity per class
    for i in xrange(number_classes):
        class_spike_activity = testing_rates[:, labels_max_voting == i]
        total_class_activity[:, i] = np.sum(class_spike_activity, axis=1)/float(class_spike_activity.shape[1])

    test_rank = np.argsort(total_class_activity, axis=1)
    predicted_max_voting = test_rank[:, -1]

    # evaluate accuracy
    difference = testing_y.reshape(-1) - predicted_max_voting
    correct = len(np.where(difference == 0)[0])
    accuracy[index, 0] = correct/float(number_samples)

    # print 'Max voting', accuracy[index, 0],

    # PREDICT LABELS CONFIDENCE VOTING

    # get neuron confidence per class
    total_class_activity = np.dot(testing_rates, labels_confidence_voting.transpose())
    predicted_confidence_voting = np.argsort(total_class_activity, axis=1)[:, -1]

    # evaluate accuracy
    difference = testing_y.reshape(-1) - predicted_confidence_voting
    correct = len(np.where(difference == 0)[0])
    accuracy[index, 1] = correct/float(number_samples)

    # print 'Confidence voting', accuracy[index, 1]

print accuracy
print 'max voting', accuracy.mean(axis=0)[0], '+/-', accuracy.std(axis=0)[0]
print 'confidence voting', accuracy.mean(axis=0)[1], '+/-', accuracy.std(axis=0)[1]
