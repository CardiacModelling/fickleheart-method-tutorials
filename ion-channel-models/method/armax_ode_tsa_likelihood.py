from __future__ import print_function
import pints
import numpy as np
import copy

class DiscrepancyLogLikelihood(pints.ProblemLogLikelihood):
    """
    This class defines a custom loglikelihood which implements a
    discrepancy model where the noise follows an ARMA(p,q) process
    """
    def __init__(self, problem, armax_result_obj, transparams=False,
            temperature=None):
        super(DiscrepancyLogLikelihood, self).__init__(problem)

        self._no = problem.n_outputs()
        self._np = problem.n_parameters()
        self._armax_result = copy.deepcopy(armax_result_obj)
        self._nds = len(self._armax_result.params) - 1
        self._n_parameters = self._np + self._nds
        self._nt = problem.n_times() 
        self._temperature = temperature
        self._transparams = transparams
        self._armax_result.model.endog = np.copy(problem._values)

    def __call__(self, x):
        armax_params = x[-self._nds:]
        armax_params = np.append(1.0, armax_params)
        model_params = x[:-self._nds]
        ion_current = self._problem.evaluate(model_params) 

        self._armax_result.model.transparams = self._transparams
        self._armax_result.model.exog = ion_current[:,None]

        try:
            ll = self._armax_result.model.loglike_kalman(armax_params)
            assert(np.sum(self._armax_result.model.exog) \
                    == np.sum(ion_current))
        except:
            import sys
            print(sys.exc_info()[0], "occured.")  # not sure what exception...
            return -1. * float('inf')

        if self._temperature is not None:
            ll = self._temperature * ll

        return ll

