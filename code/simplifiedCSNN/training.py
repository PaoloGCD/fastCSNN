'''
Training, simplified Competitive Spiking Neural Network with 400 integrate and fire spiking neurons, trace based STDP,
input weight normalization, direct inhibition, no resting time
Built on Peter Diehl's implementation https://github.com/peter-u-diehl/stdp-mnist.git

@author: Paolo G. Cachi
'''


import sys
import time
import numpy as np
import cv2

import brian_no_units
import brian as b
from brian import ms


experiment_number = 0
blurring_number = 5
class_number = -1
experiment_path = './results/'

if len(sys.argv) == 4:
    experiment_number = int(sys.argv[1])
    blurring_number = int(sys.argv[2])
    class_number = int(sys.argv[3])
    print 'EXPERIMENT', experiment_number

experiment_description = '''
Simplified CSNN
Sample-base initialization blurring %d
Neurons 400
No resting time
Normalization
Initial theta 20 mV
Tau theta 3e6 ms
Theta plus 0.05 mV
''' % blurring_number

description_file = open(experiment_path + 'result' + str(experiment_number) + "/description.txt", "w+")
description_file.write(experiment_description)
description_file.close()

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


# LOAD TRAINING SET
training_data_path = "./data/"

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
neuron_groups['spiking'].theta = 20 * b.mV


# DEFINE CONNECTIONS
connections = {}

# sensory -> spiking neurons (random initialization)
# initial_weights = np.random.random((input_neurons, excitatory_neurons))
# initial_weights += 0.01
# initial_weights *= 0.3

# sensory -> spiking neurons (sample based initialization)
initial_weights = np.copy(training_x)

# select class
if class_number >= 0:
    initial_weights = initial_weights[(training_y == class_number).reshape(-1), :]
initial_weights = initial_weights[-spiking_neurons:]

# apply blurring
if blurring_number > 0:
    for i in range(spiking_neurons):
        blur = cv2.blur(initial_weights[i].reshape(28, 28), (blurring_number, blurring_number))
        initial_weights[i] = blur.reshape(1, sensory_neurons)
initial_weights = initial_weights.transpose()

# create connection
delay_input_excitatory = (0 * b.ms, 10 * b.ms)
connections['input'] = b.Connection(neuron_groups['poisson'], neuron_groups['spiking'], structure='dense',
                                    state='ge', delay=True, max_delay=delay_input_excitatory[1])
connections['input'].connect(neuron_groups['poisson'], neuron_groups['spiking'], initial_weights,
                             delay=delay_input_excitatory)

# lateral inhibition
weight_matrix_inhibitory = np.ones(spiking_neurons) - np.identity(spiking_neurons)
weight_matrix_inhibitory *= 17.0

connections['inhibitory'] = b.Connection(neuron_groups['spiking'], neuron_groups['spiking'], structure='dense',
                                         state='gi')
connections['inhibitory'].connect(neuron_groups['spiking'], neuron_groups['spiking'], weight_matrix_inhibitory)


# DEFINE STDP
stdp_connections = {}

tc_pre_ee = 20 * b.ms
tc_post_1_ee = 20 * b.ms
tc_post_2_ee = 40 * b.ms
nu_ee_pre = 0.0001
nu_ee_post = 0.01

stdp_eqs = '''
                post2before                             : 1.0
                dpre/dt = -pre/(tc_pre_ee)              : 1.0
                dpost1/dt = -post1/(tc_post_1_ee)       : 1.0
                dpost2/dt = -post2/(tc_post_2_ee)       : 1.0
           '''

stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post1'
stdp_post_ee = 'post2before = post2; w += nu_ee_post * pre * post2before; post1 = 1.; post2 = 1.'

wmax_ee = 1.0

stdp_connections['input'] = b.STDP(connections['input'], eqs=stdp_eqs, pre=stdp_pre_ee, post=stdp_post_ee,
                                   wmin=0., wmax=wmax_ee)


# RUN TRAINING
number_epochs = 1
number_samples = training_x.shape[0]
total_samples = number_epochs * number_samples

input_intensity = 2.0
default_input_intensity = input_intensity

total_connection_weight = 78.

neuron_groups['poisson'].rate = 0
b.run(0 * b.ms)

spike_counter = b.SpikeCounter(neuron_groups['spiking'])
previous_spike_count = np.zeros(spiking_neurons)
result_spike_activity = np.zeros((total_samples, spiking_neurons))

total_start = time.time()
start = time.time()
i = 0
while i < total_samples:

    # normalize input connection weights
    total_input_weight_neuron = np.sum(connections['input'].W, axis=0)
    connections['input'].W *= total_connection_weight / total_input_weight_neuron

    # present one image
    neuron_groups['poisson'].rate = training_x[i % number_samples, :] * input_intensity

    b.run(input_image_time)

    # evaluate neuron activity
    current_spike_count = spike_counter.count - previous_spike_count
    previous_spike_count = np.copy(spike_counter.count)

    number_spikes = np.sum(current_spike_count)
    # print i, number_spikes

    if number_spikes < 5:
        input_intensity += 1
    else:
        result_spike_activity[i, :] = current_spike_count

        # print training time
        if i % 10 == 0 and i > 0:
            end = time.time()
            firing_rate = result_spike_activity[i-9:i+1, :].sum()
            print 'run:', i, 'of', total_samples, ', time: %.5f' % (end - start), ', firing rate: %5.1f' % firing_rate
            start = time.time()

        # jump to next sample
        input_intensity = default_input_intensity
        i += 1

print 'Total required time:', time.time() - total_start

# save results
np.save(experiment_path + 'result' + str(experiment_number) + '/training_delay', connections['input'].delay)
np.save(experiment_path + 'result' + str(experiment_number) + '/training_weights', connections['input'].W)
np.save(experiment_path + 'result' + str(experiment_number) + '/training_theta', neuron_groups['spiking'].theta)

np.save(experiment_path + 'result' + str(experiment_number) + '/training_rates', result_spike_activity)
