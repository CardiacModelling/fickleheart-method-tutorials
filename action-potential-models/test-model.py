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
model_ohara = m.Model('./mmt-model-files/ohara-2011.mmt')
model_ohara.set_name('ohara-2011')

# Default stimuli
times = np.linspace(0, 1000, 5000)

sim_tnnp = model_tnnp.simulate(np.ones(model_tnnp.dimension()), times)
sim_tnnpw = model_tnnpw.simulate(np.ones(model_tnnpw.dimension()), times)
sim_ohara = model_ohara.simulate(np.ones(model_ohara.dimension()), times)

for _ in range(10):
    assert(np.all(np.abs(
                model_tnnp.simulate(np.ones(model_tnnp.dimension()), times)
                - sim_tnnp) < 1e-6))
    assert(np.all(np.abs(
                model_tnnpw.simulate(np.ones(model_tnnpw.dimension()), times)
                - sim_tnnpw) < 1e-6))
    assert(np.all(np.abs(
                model_ohara.simulate(np.ones(model_ohara.dimension()), times)
                - sim_ohara) < 1e-6))

# and have a look at it
plt.figure()
plt.plot(times, sim_tnnp, label=model_tnnp.name())
plt.plot(times, sim_tnnpw, label=model_tnnpw.name())
plt.plot(times, sim_ohara, label=model_ohara.name())
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
if savefig:
    plt.savefig('%s/test-model-defaultstim.png'%savedir)
else:
    plt.show()

del(model_tnnp, model_tnnpw, model_ohara)


# Input stimuli
np.random.seed(101)
random_stim = [(0, 50)]
for r in np.random.uniform(250, 700, size=10):
    random_stim.append((1, 5))  # stim duration use default
    random_stim.append((0, r))
random_stim.append((1, 5))
random_stim.append((0, 1000))

times = np.arange(0, np.sum(np.asarray(random_stim)[:, 1]), 0.2)

model_tnnp = m.Model('./mmt-model-files/tnnp-2004.mmt',
        stim_seq=random_stim)
model_tnnp.set_name('tnnp-2004')
model_tnnpw = m.Model('./mmt-model-files/tnnp-2004-w.mmt',
        stim_seq=random_stim)
model_tnnpw.set_name('tnnp-w-2004')
model_ohara = m.Model('./mmt-model-files/ohara-2011.mmt',
        stim_seq=random_stim)
model_ohara.set_name('ohara-2011')

sim_tnnp = model_tnnp.simulate(np.ones(model_tnnp.dimension()), times)
sim_tnnpw = model_tnnpw.simulate(np.ones(model_tnnpw.dimension()), times)
sim_ohara = model_ohara.simulate(np.ones(model_ohara.dimension()), times)

for _ in range(10):
    assert(np.all(np.abs(
                model_tnnp.simulate(np.ones(model_tnnp.dimension()), times)
                - sim_tnnp) < 1e-6))
    assert(np.all(np.abs(
                model_tnnpw.simulate(np.ones(model_tnnpw.dimension()), times)
                - sim_tnnpw) < 1e-6))
    assert(np.all(np.abs(
                model_ohara.simulate(np.ones(model_ohara.dimension()), times)
                - sim_ohara) < 1e-6))

# and have a look at it
plt.figure()
plt.plot(times, sim_tnnp, label=model_tnnp.name())
plt.plot(times, sim_tnnpw, label=model_tnnpw.name())
plt.plot(times, sim_ohara, label=model_ohara.name())
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
if savefig:
    plt.savefig('%s/test-model-randstim.png'%savedir)
else:
    plt.show()
