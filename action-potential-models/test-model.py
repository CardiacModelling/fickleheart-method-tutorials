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
model_fink = m.Model('./mmt-model-files/fink-2008.mmt')
model_fink.set_name('fink-2008')
model_ohara = m.Model('./mmt-model-files/ohara-2011.mmt')
model_ohara.set_name('ohara-2011')

times = np.linspace(0, 1000, 5000)

sim_tnnp = model_tnnp.simulate(np.ones(model_tnnp.dimension()), times)
sim_fink = model_fink.simulate(np.ones(model_fink.dimension()), times)
sim_ohara = model_ohara.simulate(np.ones(model_ohara.dimension()), times)

# and have a look at it
plt.figure()
plt.plot(times, sim_tnnp, label=model_tnnp.name())
plt.plot(times, sim_fink, label=model_fink.name())
plt.plot(times, sim_ohara, label=model_ohara.name())
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
if savefig:
    plt.savefig('%s/test-model.png'%savedir)
else:
    plt.show()
