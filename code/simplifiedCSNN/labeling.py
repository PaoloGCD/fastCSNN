'''
Labeling, simplified Competitive Spiking Neural Network with 400 integrate and fire spiking neurons, trace based STDP,
input weight normalization, direct inhibition, no resting time
Built on Peter Diehl's implementation https://github.com/peter-u-diehl/stdp-mnist.git

@author: Paolo G. Cachi
'''


import time
import sys
import numpy as np

import brian_no_units
import brian as b
from brian import ms


experiment_number = 0
experiment_path = '../../result/'

if len(sys.argv) == 2:
    experiment_number = int(sys.argv[1])
    print 'EXPERIMENT', experiment_number


b.set_global_preferences(
    defaultclock=b.Clock(dt=0.1 * b.ms),  # The default clock to use if none is provided.
    useweave=True,  # Defines whether or not functions should use inlined compiled C code where defined.
    gcc_options=['-march=native'],  # Defines the compiler switches passed to the gcc compiler.
    usecodegen=True,  # Whether or not to use experimental code generation support.
    usecodegenweave=True,  # Whether or not to use C with experimental code generation support.
    usecodegenstateupdate=True,  # Whether or not to use experimental code generation support on state updaters.
    usecodegenthreshold=False,  # Whether or not to use experimental code generation support on thresholds.
    usenewpropagate=True,  # Whether or not to use experimental new C propagation functions.
    usecstdp=True,  # Whether or not to use experimental new C STDP.
)


# SYSTEM PARAMETERS
sensory_neurons = 784
spiking_neurons = 400
inhibitory_neurons = 400

input_image_time = 350 * b.ms


# LOAD DATA
load_theta = np.load(experiment_path + 'result' + str(experiment_number) + '/training_theta.npy')
load_weights = np.load(experiment_path + 'result' + str(experiment_number) + '/training_weights.npy')
load_delay = np.load(experiment_path + 'result' + str(experiment_number) + '/training_delay.npy')


# LOAD TRAINING SET
training_data_path = "../../data/"

training_x = np.load(training_data_path + 'MNIST-training-samples.npy')
training_x = training_x.astype('float')/8.0
training_x = training_x.reshape(-1, sensory_neurons)

training_y = np.load(training_data_path + 'MNIST-training-labels.npy')


# DEFINE SENSORY GROUP
neuron_groups = {'poisson': b.PoissonGroup(sensory_neurons, 0)}


# DEFINE SPIKING GROUP

# model equations
v_rest_e = -65. * b.mV
v_reset_e = -65. * b.mV
v_threshold_e = -52. * b.mV

tau_v = 100 * b.ms
tau_ge = 1.0 * b.ms
tau_gi = 2.0 * b.ms
tau_theta = 3e6 * b.ms

time_refractory_e = 5. * b.ms

neuron_eqs_e = '''
        dv/dt = ((v_rest_e-v) + ge*-v + gi*(-100.*mV-v)) / (tau_v)  : volt
        dge/dt = -ge/tau_ge                                         : 1
        dgi/dt = -gi/tau_gi                                         : 1
        dtheta/dt = -theta/(tau_theta)                              : volt
        dtimer/dt = 1                                               : ms
        '''

# reset equations
theta_plus_e = 0.05 * b.mV
reset_eqs_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'

# threshold equations
offset = 20 * b.mV
threshold_eqs_e = '(v>(theta - offset + v_threshold_e)) * (timer>time_refractory_e)'

# group instantiation
neuron_groups['spiking'] = b.NeuronGroup(N=spiking_neurons, model=neuron_eqs_e, threshold=threshold_eqs_e,
                                         refractory=time_refractory_e, reset=reset_eqs_e, compile=True, freeze=True)

neuron_groups['spiking'].v = v_rest_e
neuron_groups['spiking'].theta = load_theta


# DEFINE CONNECTIONS
connections = {}

# sensory -> spiking neurons
weight_matrix_input = load_weights

delay_input_excitatory = (0 * b.ms, 10 * b.ms)

connections['input'] = b.Connection(neuron_groups['poisson'], neuron_groups['spiking'], structure='dense',
                                    state='ge', delay=True, max_delay=delay_input_excitatory[1])
connections['input'].connect(neuron_groups['poisson'], neuron_groups['spiking'], weight_matrix_input,
                             delay=delay_input_excitatory)

connections['input'].delay[:, :] = load_delay

# lateral inhibition
weight_matrix_inhibitory = np.ones(spiking_neurons) - np.identity(spiking_neurons)
weight_matrix_inhibitory *= 17.0

connections['inhibitory'] = b.Connection(neuron_groups['spiking'], neuron_groups['spiking'], structure='dense',
                                         state='gi')
connections['inhibitory'].connect(neuron_groups['spiking'], neuron_groups['spiking'], weight_matrix_inhibitory)


# RUN LABELING
number_samples = training_x.shape[0]

input_intensity = 2.0
default_input_intensity = input_intensity

neuron_groups['poisson'].rate = 0
b.run(0 * b.ms)

spike_counter = b.SpikeCounter(neuron_groups['spiking'])
previous_spike_count = np.zeros(spiking_neurons)
result_spike_activity = np.zeros((number_samples, spiking_neurons))

total_start = time.time()
start = time.time()

i = 0
while i < number_samples:

    # present one image
    neuron_groups['poisson'].rate = training_x[i, :] * input_intensity

    b.run(input_image_time)

    # evaluate neuron activity
    current_spike_count = spike_counter.count - previous_spike_count
    previous_spike_count = np.copy(spike_counter.count)

    number_spikes = np.sum(current_spike_count)
    # print i, number_spikes

    if number_spikes < 5:
        input_intensity += 1
    else:
        if i % 10 == 0 and i > 0:
            end = time.time()
            print 'runs done:', i, 'of', int(number_samples), ', required time:', end - start
            start = time.time()

        result_spike_activity[i, :] = current_spike_count

        input_intensity = default_input_intensity
        i += 1

# save results
print 'Total required time:', time.time() - total_start
np.save(experiment_path + 'result' + str(experiment_number) + '/labeling_rates', result_spike_activity)


# GET NEURON LABELS

number_classes = 10
total_class_activity = np.zeros((number_classes, spiking_neurons))

# get neuron activity per class
for i in xrange(number_classes):
    class_spike_activity = result_spike_activity[training_y[:number_samples].reshape(-1) == i, :]
    total_class_activity[i, :] = np.sum(class_spike_activity, axis=0)/float(class_spike_activity.shape[0])

# get max class activity
class_rank = np.argsort(total_class_activity, axis=0)
neuron_labels = class_rank[-1, :]

print neuron_labels

np.save(experiment_path + 'result' + str(experiment_number) + '/labeling_labels', neuron_labels)
