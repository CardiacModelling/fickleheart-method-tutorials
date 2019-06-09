from __future__ import print_function
import numpy as np

import pints
import myokit
import myokit.pacing as pacing

###############################################################################
## Defining Model
###############################################################################

vhold = -80  # mV
# Default time
DT = 2.0e-1  # ms; maybe not use this...


#
# Time out handler
#
class Timeout(myokit.ProgressReporter):
    """
    A :class:`myokit.ProgressReporter` that halts the simulation after
    ``max_time`` seconds.
    """
    def __init__(self, max_time):
        self.max_time = float(max_time)
    def enter(self, msg=None):
        self.b = myokit.Benchmarker()
    def exit(self):
        pass
    def update(self, progress):
        return self.b.time() < self.max_time


#
# Create ForwardModel
#
class Model(pints.ForwardModel):
    
    def __init__(self, model_file, variables, current_readout, set_ion,
                 temperature, transform=None, max_evaluation_time=5):
        """
        # model_file: mmt model file for myokit; main units: mV, ms, pA
        # variables: myokit model variables, set model parameter.
        # current_readout: myokit model component, set output for method
        #                  simulate().
        # set_ion: myokit model ion concentration clamp, if any; format:
        #          {'ion_conc1': value1, ...}
        # temperature: temperature of the experiment to be set (in K).
        # transform: transform search space parameters to model parameters.
        # max_evaluation_time: maximum time (in second) allowed for one
        #                      simulate() call.
        """

        # maximum time allowed
        self.max_evaluation_time = max_evaluation_time
        
        # Load model
        model = myokit.load_model(model_file)
        # Set temperature
        model.get('phys.T').set_rhs(temperature)
        
        # Set parameters and readout of simulate()
        self.parameters = variables
        self._readout = current_readout
        assert(len(self._readout) > 0)
        
        # Set up voltage clamp
        for ion_var, ion_conc in set_ion.items():
            v = model.get(ion_var)
            if v.is_state():
                v.demote()
            v.set_rhs(ion_conc)
            
        # Detach voltage for voltage clamp(?)
        model_v = model.get('membrane.V')
        if model_v.is_state():
            model_v.demote()
        model_v.set_rhs(vhold)
        model.get('engine.pace').set_binding(None)
        model_v.set_binding('pace')

        # 1. Create pre-pacing protocol
        protocol = pacing.constant(vhold)
        # Create pre-pacing simulation
        self.simulation1 = myokit.Simulation(model, protocol)
        
        # 2. Create specified protocol place holder
        self.simulation2 = myokit.Simulation(model)
        p = [vhold, 0.1] * 3
        self.set_voltage_protocol(p)
        
        self.simulation2.set_tolerance(1e-10, 1e-12)
        self.simulation2.set_max_step_size(1e-2)  # ms
        
        # Keep model
        self._model = model

        self.transform = transform
        self.default_init_state = self.simulation1.state()
        self.init_state = self.default_init_state
        self.set_continue_simulate(False)

    def n_parameters(self):
        # n_parameters() method for Pints
        return len(self.parameters)

    def set_init_state(self, v):
        self.init_state = v

    def current_state(self):
        return self.simulation2.state()

    def set_continue_simulate(self, v=False):
        self._continue_simulate = v
        
    def set_voltage_protocol(self, p):
        # Assume protocol p is
        # [step_1_voltage, step_1_duration, step_2_voltage, ...]
        protocol = myokit.Protocol()
        for i in range(len(p) // 2):
            protocol.add_step(p[2 * i], p[2 * i + 1])
        self.simulation2.set_protocol(protocol)
        del(protocol)

    def set_fixed_form_voltage_protocol(self, v, t):
        # v, t: voltage, time to be set in ms, mV
        self.simulation2.set_fixed_form_protocol(
            t, v  # ms, mV
        )

    def current_list(self):
        return self._readout

    def simulate(self, parameters, times, read_log=None):
        # simulate() method for Pints

        # Update model parameters
        if self.transform is not None:
            parameters = self.transform(parameters)

        for i, name in enumerate(self.parameters):
            self.simulation1.set_constant(name, parameters[i])
            self.simulation2.set_constant(name, parameters[i])

        # Reset to ensure each simulate has same init condition
        self.simulation2.reset()
        self.simulation2.set_state(self.init_state)

        if not self._continue_simulate:
            self.simulation1.reset()
            self.simulation1.set_state(self.init_state)
            try:
                self.simulation1.pre(100e3)
            except (myokit.SimulationError, myokit.SimulationCancelledError):
                # return float('inf')
                return np.full(times.shape, float('inf'))
            self.simulation2.set_state(self.simulation1.state())
        
        if read_log is None:
            to_read = self._readout
        else:
            to_read = read_log

        # Run!
        try:
            p = Timeout(self.max_evaluation_time)
            d = self.simulation2.run(np.max(times)+0.02e3,
                log_times = times,
                log = to_read,
                #log_interval = 0.025
                progress=p,
                ).npview()
            del(p)
        except (myokit.SimulationError, myokit.SimulationCancelledError):
            # return float('inf')
            return np.full(times.shape, float('inf'))

        # Return all lump currents
        if read_log is None:
            o = np.zeros(times.shape)
            for i in to_read:
                o += d[i]
            return o
        else:
            return d

    def voltage(self, times, parameters=None):
        # Return voltage protocol

        if parameters is not None:
            # Update model parameters
            if self.transform is not None:
                parameters = self.transform(parameters)

            for i, name in enumerate(self.parameters):
                self.simulation1.set_constant(name, parameters[i])
                self.simulation2.set_constant(name, parameters[i])

        # Run
        self.simulation1.reset()
        self.simulation2.reset()
        self.simulation1.set_state(self.init_state)
        self.simulation2.set_state(self.init_state)
        try:
            self.simulation1.pre(100e3)
            self.simulation2.set_state(self.simulation1.state())
            d = self.simulation2.run(np.max(times)+0.02e3,
                log_times = times,
                log = ['membrane.V'],
                #log_interval = 0.025
                ).npview()
        except myokit.SimulationError:
            # return float('inf')
            return np.full(times.shape, float('inf'))

        # Return
        return d['membrane.V'] 
    
    def parameter(self):
        # return the name of the parameters
        return self.parameters

