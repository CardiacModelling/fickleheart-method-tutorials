import numpy as np

# Output name
save_name = 'model_A'

# myokit mmt file path from `ion-channel-tutorial`
model_file = './mmt-model-files/model-A.mmt'

# myokit current names that can be observed
# Assume only the sum of all current can be observed if multiple currents
current_list = ['ikr.IKr']

# myokit variable names
# All the parameters to be inferred
parameters = [
    'ikr.g',
    'ikr.p1', 'ikr.p2', 'ikr.p3', 'ikr.p4',
    'ikr.p5', 'ikr.p6', 'ikr.p7', 'ikr.p8',
    ]

# Indicating which current a parameter is belonged to
var_list = {'ikr.g': 'ikr.IKr',
            'ikr.p1': 'ikr.IKr',
            'ikr.p2': 'ikr.IKr',
            'ikr.p3': 'ikr.IKr',
            'ikr.p4': 'ikr.IKr',
            'ikr.p5': 'ikr.IKr',
            'ikr.p6': 'ikr.IKr',
            'ikr.p7': 'ikr.IKr',
            'ikr.p8': 'ikr.IKr',
        }

# Prior knowledge of the model parameters
base_param = np.array([ # Beattie et al. 2018 cell5
    0.1524 * 1e3,  # pA/mV
    2.26e-4,
    0.0699,
    3.45e-5,
    0.05462,
    0.0873,
    8.91e-3,
    5.15e-3,
    0.03158,
    ]) # mV, ms

# Clamp ion; assumption in voltage clamp
ions_conc = {
    'potassium.Ki': 120,
    'potassium.Ko': 5,
    }  # mM

# Prior
import sys
sys.path.append('../method')  # where priors module is
from priors import ModelALogPrior as LogPrior

# Temperature of the experiment
temperature = 37.0  # oC

