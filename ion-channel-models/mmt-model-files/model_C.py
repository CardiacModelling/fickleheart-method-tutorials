import numpy as np

# Output name
save_name = 'model_C'

# myokit mmt file path from `ion-channel-tutorial`
model_file = './mmt-model-files/model-C.mmt'

# myokit current names that can be observed
# Assume only the sum of all current can be observed if multiple currents
current_list = ['ikr.IKr']

# myokit variable names
# All the parameters to be inferred
parameters = [
    'ikr.g',
    'ikr.p1', 'ikr.p2', 'ikr.p3', 'ikr.p4',
    'ikr.p5', 'ikr.p6', 'ikr.p7', 'ikr.p8',
    'ikr.p9', 'ikr.p10', 'ikr.p11', 'ikr.p12',
    'ikr.p13', 'ikr.p14', 'ikr.p15', 'ikr.p16',
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
            'ikr.p9': 'ikr.IKr',
            'ikr.p10': 'ikr.IKr',
            'ikr.p11': 'ikr.IKr',
            'ikr.p12': 'ikr.IKr',
            'ikr.p13': 'ikr.IKr',
            'ikr.p14': 'ikr.IKr',
            'ikr.p15': 'ikr.IKr',
            'ikr.p16': 'ikr.IKr',
        }

# Prior knowledge of the model parameters
base_param = np.array([ # Mazhari et al. 2001
    0.1524 * 1e3,  # pA/mV
    0.0218,
    0.0262,
    0.0009,
    0.0269,
    0.0266,
    0.1348,
    0.0622,
    0.0120,
    0.0059,
    0.0443,
    0.0069,
    0.0272,
    0.0227,
    0.0431,
    0.0000129,
    0.00000271,
    ]) # mV, ms

# Clamp ion; assumption in voltage clamp
ions_conc = {
    'potassium.Ki': 120,
    'potassium.Ko': 5,
    }  # mM

# Prior
import sys
sys.path.append('../method')  # where priors module is
from priors import ModelCLogPrior as LogPrior

# Temperature of the experiment
temperature = 24.0  # oC

