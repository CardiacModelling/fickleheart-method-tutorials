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

savedir = './fig'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Get all input variables
import importlib
sys.path.append('./mmt-model-files')
info_ida = 'model_A'
infoa = importlib.import_module(info_ida)
info_idb = 'model_B'
infob = importlib.import_module(info_idb)
info_idc = 'model_C'
infoc = importlib.import_module(info_idc)

modela = m.Model(infoa.model_file,
        variables=infoa.parameters,
        current_readout=infoa.current_list,
        set_ion=infoa.ions_conc,
        temperature=273.15 + infoa.temperature,  # K
        )
modelb = m.Model(infob.model_file,
        variables=infob.parameters,
        current_readout=infob.current_list,
        set_ion=infob.ions_conc,
        temperature=273.15 + infob.temperature,  # K
        )
modelc = m.Model(infoc.model_file,
        variables=infoc.parameters,
        current_readout=infoc.current_list,
        set_ion=infoc.ions_conc,
        temperature=273.15 + infoc.temperature,  # K
        )

#
# Activation step protocol
#
# Load protocol
test_prt = [-80, 200, 20, 500, -40, 500, -80, 200]
dt = 0.1
test_t = np.arange(0, np.sum(test_prt[1::2]), dt)

# Update protocol
modela.set_voltage_protocol(test_prt)
modelb.set_voltage_protocol(test_prt)
modelc.set_voltage_protocol(test_prt)

# Plot
plt.figure(figsize=(8, 6))
plt.subplot(211)
plt.plot(test_t, modela.voltage(test_t), c='#7f7f7f')
plt.plot(test_t, modelb.voltage(test_t), c='#7f7f7f')
plt.plot(test_t, modelc.voltage(test_t), c='#7f7f7f')
plt.ylabel('Voltage (mV)')

plt.subplot(212)
plt.plot(test_t, modela.simulate(infoa.base_param, test_t), label='Model A')
plt.plot(test_t, modelb.simulate(infob.base_param, test_t), label='Model B')
plt.plot(test_t, modelc.simulate(infoc.base_param, test_t), label='Model C')
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Current (pA)')

plt.subplots_adjust(hspace=0)
plt.savefig('%s/test-models-activation.png' % (savedir), bbox_inches='tight')
plt.close()

#
# Sine wave protocol
#
# Load protocol
test_prt = np.loadtxt('./protocol-time-series/sinewave-ramp.csv', skiprows=1,
        delimiter=',')
test_t = test_prt[:, 0]
test_prt = test_prt[:, 1]

# Update protocol
modela.set_fixed_form_voltage_protocol(test_prt, test_t)
modelb.set_fixed_form_voltage_protocol(test_prt, test_t)
modelc.set_fixed_form_voltage_protocol(test_prt, test_t)

# Plot
plt.figure(figsize=(8, 6))
plt.subplot(211)
plt.plot(test_t, modela.voltage(test_t), c='#7f7f7f')
plt.plot(test_t, modelb.voltage(test_t), c='#7f7f7f')
plt.plot(test_t, modelc.voltage(test_t), c='#7f7f7f')
plt.ylabel('Voltage (mV)')

plt.subplot(212)
plt.plot(test_t, modela.simulate(infoa.base_param, test_t), label='Model A')
plt.plot(test_t, modelb.simulate(infob.base_param, test_t), label='Model B')
plt.plot(test_t, modelc.simulate(infoc.base_param, test_t), label='Model C')
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Current (pA)')

plt.subplots_adjust(hspace=0)
plt.savefig('%s/test-models-sinewave.png' % (savedir), bbox_inches='tight')
plt.close()

#
# AP protocols
#
# Load protocol
test_prt = np.loadtxt('./protocol-time-series/ap.csv', skiprows=1,
        delimiter=',')
test_t = test_prt[:, 0]
test_prt = test_prt[:, 1]

# Update protocol
modela.set_fixed_form_voltage_protocol(test_prt, test_t)
modelb.set_fixed_form_voltage_protocol(test_prt, test_t)
modelc.set_fixed_form_voltage_protocol(test_prt, test_t)

# Plot
plt.figure(figsize=(8, 6))
plt.subplot(211)
plt.plot(test_t, modela.voltage(test_t), c='#7f7f7f')
plt.plot(test_t, modelb.voltage(test_t), c='#7f7f7f')
plt.plot(test_t, modelc.voltage(test_t), c='#7f7f7f')
plt.ylabel('Voltage (mV)')

plt.subplot(212)
plt.plot(test_t, modela.simulate(infoa.base_param, test_t), label='Model A')
plt.plot(test_t, modelb.simulate(infob.base_param, test_t), label='Model B')
plt.plot(test_t, modelc.simulate(infoc.base_param, test_t), label='Model C')
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Current (pA)')

plt.subplots_adjust(hspace=0)
plt.savefig('%s/test-models-ap.png' % (savedir), bbox_inches='tight')
plt.close()
