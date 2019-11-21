import sys
import numpy as np
import tensorflow as tf


experiment_number = 0
experiment_path = '../../result/'

if len(sys.argv) == 2:
    experiment_number = int(sys.argv[1])


# LOAD TRAINING LABELS
training_data_path = "../../data/"
training_x = np.load(training_data_path + 'MNIST-training-samples.npy')
training_y = np.load(training_data_path + 'MNIST-training-labels.npy')


# LOAD TESTING LABELS
testing_data_path = "../../data/"
testing_x = np.load(testing_data_path + 'MNIST-testing-samples.npy')
testing_y = np.load(testing_data_path + 'MNIST-testing-labels.npy')


# LOAD RATES
labeling_rates = np.load(experiment_path + 'result' + str(experiment_number) + '/labeling_rates.npy')
testing_rates = np.load(experiment_path + 'result' + str(experiment_number) + '/testing_rates.npy')

# print labeling_rates.shape
# print testing_rates.shape

val_hist = np.zeros((10, 10))
for i in range(10):

    print i

    # CREATE CLASSIFICATION MODEL
    model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Dense(400, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(200, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    # FIT MODEL
    batch_size = 32
    epochs = 10

    training_samples = labeling_rates
    training_labels = training_y.astype('float').reshape(60000, -1)

    validating_samples = testing_rates
    validating_labels = testing_y.astype('float').reshape(10000, -1)

    # training_samples = training_x.astype('float').reshape(60000, -1)
    # training_labels = training_y.astype('float').reshape(60000, -1)
    #
    # validating_samples = testing_x.astype('float').reshape(10000, -1)
    # validating_labels = testing_y.astype('float').reshape(10000, -1)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(x=training_samples, y=training_labels, batch_size=batch_size, epochs=epochs, verbose=1,
                     validation_data=(validating_samples, validating_labels))

    val_hist[i, :] = hist.history['val_acc']

print val_hist
print 'mean', val_hist.mean(axis=0)
print 'std', val_hist.std(axis=0)

np.save(experiment_path + 'result' + str(experiment_number) + '/nn_validation', val_hist)
