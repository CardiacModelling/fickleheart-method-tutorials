#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append('./method')
import os
import numpy as np
savefig = True
if savefig:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import model as m

savedir = './fig'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

model_tnnp = m.Model('./mmt-model-files/tnnp-2004.mmt')
model_tnnp.set_name('tnnp-2004')
model_tnnpw = m.Model('./mmt-model-files/tnnp-2004-w.mmt')
model_tnnpw.set_name('tnnp-w-2004')
model_fink = m.Model('./mmt-model-files/fink-2008.mmt')
model_fink.set_name('fink-2008')

# Default stimuli
times = np.linspace(0, 1000, 5000)

sim_tnnp = model_tnnp.simulate(np.ones(model_tnnp.n_parameters()), times)
sim_tnnpw = model_tnnpw.simulate(np.ones(model_tnnpw.n_parameters()), times)
sim_fink = model_fink.simulate(np.ones(model_fink.n_parameters()), times)

for _ in range(10):
    assert(np.all(np.abs(
                model_tnnp.simulate(np.ones(model_tnnp.n_parameters()), times)
                - sim_tnnp) < 1e-6))
    assert(np.all(np.abs(
                model_tnnpw.simulate(np.ones(model_tnnpw.n_parameters()),
                    times)
                - sim_tnnpw) < 1e-6))
    assert(np.all(np.abs(
                model_fink.simulate(np.ones(model_fink.n_parameters()),
                    times)
                - sim_fink) < 1e-6))

# and have a look at it
plt.figure()
plt.plot(times, sim_tnnp, label=model_tnnp.name())
plt.plot(times, sim_tnnpw, label=model_tnnpw.name())
plt.plot(times, sim_fink, label=model_fink.name())
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
if savefig:
    plt.savefig('%s/test-model-defaultstim.png'%savedir)
else:
    plt.show()
plt.close()

del(model_tnnp, model_tnnpw, model_fink)


# Input stimuli
np.random.seed(101)
random_stim = [(0, 50)]
for r in np.random.uniform(100, 600, size=10):
    random_stim.append((1, 1))  # stim duration use default
    random_stim.append((0, r))
random_stim.append((1, 1))  # stim duration use default
random_stim.append((0, 500))

times = np.arange(0, np.sum(np.asarray(random_stim)[:, 1]), 0.2)

model_tnnp = m.Model('./mmt-model-files/tnnp-2004.mmt',
        stim_seq=random_stim)
model_tnnp.set_name('tnnp-2004')
model_tnnpw = m.Model('./mmt-model-files/tnnp-2004-w.mmt',
        stim_seq=random_stim)
model_tnnpw.set_name('tnnp-w-2004')
model_fink = m.Model('./mmt-model-files/fink-2008.mmt',
        stim_seq=random_stim)
model_fink.set_name('fink-2008')

sim_tnnp = model_tnnp.simulate(np.ones(model_tnnp.n_parameters()), times)
sim_tnnpw = model_tnnpw.simulate(np.ones(model_tnnpw.n_parameters()), times)
sim_fink = model_fink.simulate(np.ones(model_fink.n_parameters()), times)

for _ in range(10):
    assert(np.all(np.abs(
                model_tnnp.simulate(np.ones(model_tnnp.n_parameters()), times)
                - sim_tnnp) < 1e-6))
    assert(np.all(np.abs(
                model_tnnpw.simulate(np.ones(model_tnnpw.n_parameters()),
                    times)
                - sim_tnnpw) < 1e-6))
    assert(np.all(np.abs(
                model_fink.simulate(np.ones(model_fink.n_parameters()),
                    times)
                - sim_fink) < 1e-6))

# and have a look at it
plt.figure()
plt.plot(times, sim_tnnp, label=model_tnnp.name())
plt.plot(times, sim_tnnpw, label=model_tnnpw.name())
plt.plot(times, sim_fink, label=model_fink.name())
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
if savefig:
    plt.savefig('%s/test-model-randstim.png'%savedir)
else:
    plt.show()
plt.close()

