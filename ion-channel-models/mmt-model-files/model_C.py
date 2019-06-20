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
parameters = []

# Indicating which current a parameter is belonged to
var_list = {}

# Prior knowledge of the model parameters
base_param = np.array([]) # mV, ms

# Clamp ion; assumption in voltage clamp
ions_conc = {
    'potassium.Ki': 120,
    'potassium.Ko': 5,
    }  # mM

# Prior

# Temperature of the experiment
temperature = 37.0  # oC

