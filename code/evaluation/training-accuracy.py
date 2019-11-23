import sys
import numpy as np
import matplotlib.pyplot as plt


experiment_number = 0
experiment_path = './results/'

if len(sys.argv) == 2:
    experiment_number = int(sys.argv[1])


# SYSTEM PARAMETERS
poisson_neurons = 784
spiking_neurons = 400


# LOAD TRAINING LABELS
training_data_path = "./data/"
training_y = np.load(training_data_path + 'MNIST-training-labels.npy')


training_accuracy_list = []

# load experiment data
training_rates = np.load(experiment_path + '/result' + str(experiment_number) + '/training_rates.npy')


# estimate training accuracy
step = 1000
window = 1000
end = training_rates.shape[0]

number_classes = 10

temp_labels = np.zeros(spiking_neurons)
temp_class_activity = np.zeros((number_classes, spiking_neurons))
temp_neuron_activity = np.zeros((window, number_classes))
experiment_accuracy = []

number_epochs = (end+training_y.shape[0]-1)/training_y.shape[0]
stacked_labels = np.copy(training_y)
for i in xrange(1, number_epochs):
    stacked_labels = np.hstack((stacked_labels, training_y))

for i in xrange(window, end + 1, step):

    # get neuron activity per label
    for class_label in xrange(number_classes):
        label_spike_activity = training_rates[i - window:i, temp_labels == class_label]
        if label_spike_activity.shape[1] == 0:
            continue
        temp_neuron_activity[:, class_label] = np.sum(label_spike_activity, axis=1) / float(
            label_spike_activity.shape[1])

    # get max label activity
    predicted_labels = np.argsort(temp_neuron_activity, axis=1)[:, -1]

    # evaluate accuracy
    difference = stacked_labels[i - window:i].reshape(-1) - predicted_labels
    correct = len(np.where(difference == 0)[0])
    experiment_accuracy.append((i, correct / float(window)))

    # print training_accuracy
    if i > training_rates.shape[0]:
        continue

    # update labels
    for class_label in xrange(10):
        temp_spike_activity = training_rates[i - window:i, :]
        # temp_spike_activity = temp_spike_activity/temp_spike_activity.sum(axis=1).reshape((window, 1))
        class_spike_activity = temp_spike_activity[stacked_labels[i - window:i].reshape(-1) == class_label, :]
        temp_class_activity[class_label, :] = np.sum(class_spike_activity, axis=0) / float(
            class_spike_activity.shape[0])

    temp_labels = np.argsort(temp_class_activity, axis=0)[-1, :]

# save result
training_accuracy = np.array(experiment_accuracy)
np.save(experiment_path + 'result' + str(experiment_number) + '/training_accuracy', training_accuracy)

print training_accuracy

# plot training accuracy
training_accuracy_times = training_accuracy[:, 0]
training_accuracy_values = training_accuracy[:, 1]

plt.plot(training_accuracy_times[1:], training_accuracy_values[1:])
plt.show()
