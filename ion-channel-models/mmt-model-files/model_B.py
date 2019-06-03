import numpy as np

# Output name
save_name = 'model_B'

# myokit mmt file path from `ion-channel-tutorial`
model_file = './mmt-model-files/model-B.mmt'

# myokit current names that can be observed
# Assume only the sum of all current can be observed if multiple currents
current_list = ['ikr.IKr']

# myokit variable names
# All the parameters to be inferred
parameters = [
    'ikr.g',
    'ikr.p1', 'ikr.p2', 'ikr.p3', 'ikr.p4',
    'ikr.p5', 'ikr.p6', 'ikr.p7', 'ikr.p8',
    'ikr.p9', 'ikr.p10',
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
        }

# Prior knowledge of the model parameters
base_param = np.array([ # Oehmen et al. 2002
    0.1524 * 1e3,  # pA/mV
    0.0787,
    0.0378,
    0.0035,
    0.0252,
    0.0176,
    0.684,
    0.2977,
    0.0164,
    0.0862,
    0.0454,
    ]) # mV, ms

# Clamp ion; assumption in voltage clamp
ions_conc = {
    'potassium.Ki': 120,
    'potassium.Ko': 5,
    }  # mM

# Prior
import sys
sys.path.append('../method')  # where priors module is
from priors import ModelBLogPrior as LogPrior

# Temperature of the experiment
temperature = 24.0  # oC

