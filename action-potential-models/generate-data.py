#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append('./method')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import model as m
import protocol as p

savedir = './data'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

iid_noise_sigma = 1  # mV

np.random.seed(101)

#
# stim1hz
#
stim_seq = p.stim1hz
times = p.stim1hz_times

model_tnnp = m.Model('./mmt-model-files/tnnp-2004.mmt', stim_seq=stim_seq)
model_tnnp.set_name('tnnp-2004')

# Simulate and add noise
simulated_data = model_tnnp.simulate(np.ones(model_tnnp.dimension()), times)
for _ in range(10):
    assert(np.all(np.abs(
                model_tnnp.simulate(np.ones(model_tnnp.dimension()), times)
                - simulated_data) < 1e-6))
simulated_data += np.random.normal(0, iid_noise_sigma,
        size=simulated_data.shape)

# Plot
plt.figure(figsize=(8, 4))

plt.subplot(111)
plt.plot(times, simulated_data, label=model_tnnp.name())
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')

plt.subplots_adjust(hspace=0)
plt.savefig('%s/data-stim1hz.png' % (savedir), bbox_inches='tight')
plt.close()

# Save
np.savetxt('%s/data-stim1hz.csv' % (savedir),
        np.array([times, simulated_data]).T, delimiter=',', comments='',
        header='\"time\",\"voltage\"')

del(model_tnnp)


#
# stim2hz
#
stim_seq = p.stim2hz
times = p.stim2hz_times

model_tnnp = m.Model('./mmt-model-files/tnnp-2004.mmt', stim_seq=stim_seq)
model_tnnp.set_name('tnnp-2004')

# Simulate and add noise
simulated_data = model_tnnp.simulate(np.ones(model_tnnp.dimension()), times)
for _ in range(10):
    assert(np.all(np.abs(
                model_tnnp.simulate(np.ones(model_tnnp.dimension()), times)
                - simulated_data) < 1e-6))
simulated_data += np.random.normal(0, iid_noise_sigma,
        size=simulated_data.shape)

# Plot
plt.figure(figsize=(8, 4))

plt.subplot(111)
plt.plot(times, simulated_data, label=model_tnnp.name())
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')

plt.subplots_adjust(hspace=0)
plt.savefig('%s/data-stim2hz.png' % (savedir), bbox_inches='tight')
plt.close()

# Save
np.savetxt('%s/data-stim2hz.csv' % (savedir),
        np.array([times, simulated_data]).T, delimiter=',', comments='',
        header='\"time\",\"voltage\"')

del(model_tnnp)


#
# randstim
#
stim_seq = p.randstim
times = p.randstim_times

model_tnnp = m.Model('./mmt-model-files/tnnp-2004.mmt', stim_seq=stim_seq)
model_tnnp.set_name('tnnp-2004')

# Simulate and add noise
simulated_data = model_tnnp.simulate(np.ones(model_tnnp.dimension()), times)
for _ in range(10):
    assert(np.all(np.abs(
                model_tnnp.simulate(np.ones(model_tnnp.dimension()), times)
                - simulated_data) < 1e-6))
simulated_data += np.random.normal(0, iid_noise_sigma,
        size=simulated_data.shape)

# Plot
plt.figure(figsize=(8, 4))

plt.subplot(111)
plt.plot(times, simulated_data, label=model_tnnp.name())
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')

plt.subplots_adjust(hspace=0)
plt.savefig('%s/data-randstim.png' % (savedir), bbox_inches='tight')
plt.close()

# Save
np.savetxt('%s/data-randstim.csv' % (savedir),
        np.array([times, simulated_data]).T, delimiter=',', comments='',
        header='\"time\",\"voltage\"')

del(model_tnnp)

#
# hergblock
#
stim_seq = p.stim1hz_hergblock
times = p.stim1hz_hergblock_times

model_tnnp = m.Model('./mmt-model-files/tnnp-2004.mmt', stim_seq=stim_seq)
model_tnnp.set_name('tnnp-2004')

# Simulate and add noise
simulated_data = p.hergblock_simulate(model_tnnp,
        np.ones(model_tnnp.dimension()), times)
for _ in range(10):
    assert(np.all(np.abs(
                p.hergblock_simulate(model_tnnp,
                    np.ones(model_tnnp.dimension()), times)
                - simulated_data) < 1e-6))
simulated_data += np.random.normal(0, iid_noise_sigma,
        size=simulated_data.shape)

# Plot
plt.figure(figsize=(8, 4))

plt.subplot(111)
for b, s in zip(p.hergblock_fraction, simulated_data):
    plt.plot(times, s,
            label=model_tnnp.name() + ' hERG block %.0f' % (b * 100))
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')

plt.subplots_adjust(hspace=0)
plt.savefig('%s/data-hergblock.png' % (savedir), bbox_inches='tight')
plt.close()

# Save
np.savetxt('%s/data-hergblock.csv' % (savedir),
        np.vstack((times, simulated_data)).T, delimiter=',', comments='',
        header='\"time\"' + ',\"voltage\"' * len(p.hergblock_fraction))

del(model_tnnp)
