from __future__ import print_function
import numpy as np
import scipy.stats as stats
import pints


#
# Set up prior for Model A
#
class ModelALogPrior(pints.LogPrior):
    """
    Unnormalised prior with constraint on the rate constants.

    # Adapted from 
    https://github.com/CardiacModelling/hERGRapidCharacterisation

    # Added parameter transformation everywhere
    """
    def __init__(self, transform, inv_transform):
        super(ModelALogPrior, self).__init__()

        # Give it a big bound...
        self.lower_conductance = 1e2 * 1e-3  # pA/mV
        self.upper_conductance = 5e5 * 1e-3  # pA/mV

        self.lower_alpha = 1e-7
        self.upper_alpha = 1e3
        self.lower_beta  = 1e-7
        self.upper_beta  = 0.4

        self.lower = np.array([
            self.lower_conductance,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
        ])
        self.upper = np.array([
            self.upper_conductance,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
        ])

        self.minf = -float('inf')

        self.rmin = 1.67e-5
        self.rmax = 1000

        self.vmin = -120
        self.vmax =  60

        self.transform = transform
        self.inv_transform = inv_transform

    def n_parameters(self):
        return 8 + 1

    def __call__(self, parameters):

        debug = False
        log_det_Jac = np.sum(parameters)  # Jacobian adjustment
        parameters = self.transform(parameters)

        # Check parameter boundaries
        if np.any(parameters < self.lower):
            if debug: print('Lower')
            return self.minf
        if np.any(parameters > self.upper):
            if debug: print('Upper')
            return self.minf

        # Check rate constant boundaries
        g, p1, p2, p3, p4, p5, p6, p7, p8 = parameters[:]

        # Check forward rates
        r = p1 * np.exp(p2 * self.vmax)
        if np.any(r < self.rmin) or np.any(r > self.rmax):
            if debug: print('r1')
            return self.minf
        r = p5 * np.exp(p6 * self.vmax)
        if np.any(r < self.rmin) or np.any(r > self.rmax):
            if debug: print('r2')
            return self.minf

        # Check backward rates
        r = p3 * np.exp(-p4 * self.vmin)
        if np.any(r < self.rmin) or np.any(r > self.rmax):
            if debug: print('r3')
            return self.minf
        r = p7 * np.exp(-p8 * self.vmin)
        if np.any(r < self.rmin) or np.any(r > self.rmax):
            if debug: print('r4')
            return self.minf

        return 0 + log_det_Jac

    def _sample_partial(self, v):
        for i in range(100):
            a = np.exp(np.random.uniform(
                np.log(self.lower_alpha), np.log(self.upper_alpha)))
            b = np.random.uniform(self.lower_beta, self.upper_beta)
            r = a * np.exp(b * v)
            if r >= self.rmin and r <= self.rmax:
                return a, b
        raise ValueError('Too many iterations')

    def sample(self, n=1):
        out = np.zeros((n, self.n_parameters()))
        
        for i in range(n):
            p = np.zeros(self.n_parameters())

            # Sample forward rates
            p[1:3] = self._sample_partial(self.vmax)
            p[5:7] = self._sample_partial(self.vmax)

            # Sample backward rates
            p[3:5] = self._sample_partial(-self.vmin)
            p[7:9] = self._sample_partial(-self.vmin)

            # Sample conductance
            p[0] = np.random.uniform(
                self.lower_conductance, self.upper_conductance)

            out[i, :] = self.inv_transform(p)

        # Return
        return out


class ModelBLogPrior(pints.LogPrior):
    """
    Unnormalised prior with constraint on the rate constants.
    """
    def __init__(self, transform, inv_transform):
        super(ModelBLogPrior, self).__init__()

        # Give it a big bound...
        self.lower_conductance = 1e2 * 1e-3  # pA/mV
        self.upper_conductance = 5e5 * 1e-3  # pA/mV

        self.lower_alpha = 1e-7
        self.upper_alpha = 1e3
        self.lower_beta  = 1e-7
        self.upper_beta  = 0.4

        self.rmin = 1.67e-5
        self.rmax = 1000

        self.vmin = -120
        self.vmax =  60


        self.lower = np.array(
            [self.lower_conductance] + 
            [self.lower_alpha,
             self.lower_beta] * 2 + 
            [self.rmin] * 2 +
            [self.lower_alpha,
             self.lower_beta] * 2
            )
        self.upper = np.array(
            [self.upper_conductance] +
            [self.upper_alpha,
            self.upper_beta] * 2 +
            [self.rmax] * 2 +
            [self.upper_alpha,
            self.upper_beta] * 2
            )

        self.minf = -float('inf')

        self.transform = transform
        self.inv_transform = inv_transform

    def n_parameters(self):
        return 10 + 1

    def __call__(self, parameters):

        debug = False
        log_det_Jac = np.sum(parameters)  # Jacobian adjustment
        parameters = self.transform(parameters)

        # Check parameter boundaries
        if np.any(parameters < self.lower):
            if debug: print('Lower')
            return self.minf
        if np.any(parameters > self.upper):
            if debug: print('Upper')
            return self.minf

        for j in [1, 7]:  # 2 pairs of rate constants in the form A*exp(B*V)
            # Check forward rates
            r = parameters[j] * np.exp(parameters[j + 1] * self.vmax)
            if np.any(r < self.rmin) or np.any(r > self.rmax):
                if debug: print('r with p%s and p%s' % (j, j + 1))
                return self.minf

            # Check backward rates
            r = parameters[j + 2] * np.exp(-parameters[j + 3] * self.vmin)
            if np.any(r < self.rmin) or np.any(r > self.rmax):
                if debug: print('r with p%s and p%s' % (j + 2, j + 3))
                return self.minf

        for j in [5, 6]:  # 2 parameters are rate constants
            r = parameters[j]
            if np.any(r < self.rmin) or np.any(r > self.rmax):
                if debug: print('r with p%s' % (j))
                return self.minf

        return 0 + log_det_Jac

    def _sample_partial(self, v):
        for i in range(100):
            a = np.exp(np.random.uniform(
                np.log(self.lower_alpha), np.log(self.upper_alpha)))
            b = np.random.uniform(self.lower_beta, self.upper_beta)
            r = a * np.exp(b * v)
            if r >= self.rmin and r <= self.rmax:
                return a, b
        raise ValueError('Too many iterations')

    def sample(self, n=1):
        out = np.zeros((n, self.n_parameters()))
        
        for i in range(n):
            p = np.zeros(self.n_parameters())

            # Sample forward rates
            p[1:3] = self._sample_partial(self.vmax)
            p[7:9] = self._sample_partial(self.vmax)

            # Sample backward rates
            p[3:5] = self._sample_partial(-self.vmin)
            p[9:11] = self._sample_partial(-self.vmin)

            # rates
            p[5] = np.random.uniform(self.rmin, self.rmax)
            p[6] = np.random.uniform(self.rmin, self.rmax)

            # Sample conductance
            p[0] = np.random.uniform(
                self.lower_conductance, self.upper_conductance)

            out[i, :] = self.inv_transform(p)

        # Return
        return out


#
# Multiple priori
#
class MultiPriori(pints.LogPrior):
    """
    Combine multiple priors.
    """
    def __init__(self, priors):
        self._priors = priors
        self._n_parameters = self._priors[0].n_parameters()
        for p in self._priors:
            assert(self._n_parameters == p.n_parameters())

    def n_parameters(self):
        return self._n_parameters

    def __call__(self, x):
        t = 0
        for p in self._priors:
            t += p(x)
        return t


#
# Some useful for GP
#
class HalfNormalLogPrior(pints.LogPrior):
    """
    Half-Normal log-prior.
    """
    def __init__(self, sd, transform=False):
        super(HalfNormalLogPrior, self).__init__()
        # Check inputs
        if float(sd) <= 0:
            raise ValueError('Scale `sd` must be positive.')
        self._sd = float(sd)
        self.n_params = 1
        self.transform = transform

    def n_parameters(self):
        return self.n_params

    def __call__(self, x):
        if self.transform:
            Utx_x = np.exp(x)
            logp = stats.halfnorm.logpdf(Utx_x, scale=self._sd)
            logp += x
        else:
            logp = stats.halfnorm.logpdf(x, scale=self._sd)
        return np.sum(logp)

    def sample(self, n=1):
        if self.transform:
            sample = np.log(stats.halfnorm(scale=self._sd).rvs(n))
        else:
            sample = stats.halfnorm(scale=self._sd).rvs(n)
        return sample


class InverseGammaLogPrior(pints.LogPrior):
    """
    Inverse-Gamma log-prior.
    """
    def __init__(self, alpha, beta, transform = False):
        super(InverseGammaLogPrior, self).__init__()
        # Check inputs
        if (float(alpha) or float(beta)) <= 0:
            raise ValueError('Shape and Scale must be positive.')
        self._alpha = float(alpha)
        self._beta = float(beta)
        self.n_params = 1
        self.transform = transform

    def n_parameters(self):
        return self.n_params

    def __call__(self, x):
        if self.transform:
            Utx_x = np.exp(x)
            logp = stats.invgamma.logpdf(Utx_x, a=self._alpha,
                    scale=self._beta)
            logp += x
        else:
            logp = stats.invgamma.logpdf(x, a=self._alpha, scale=self._beta)
        return np.sum(logp)

    def sample(self, n=1):
        if self.transform:
            sample = np.log(stats.invgamma(a=self._alpha,
                    scale=self._beta).rvs(n))
        else:
            sample = stats.invgamma(a=self._alpha, scale=self._beta).rvs(n)
        return sample


#
# Some useful for ARMA
#
class ArmaLogPrior(pints.LogPrior):
    """
    Log prior for ARMA model.
    """
    def __init__(self, armax_result, left, right, scale):
        super(ArmaLogPrior, self).__init__()
        self._location  = armax_result.params[1:]
        self._scale = float(scale)
        self.a = float(left)
        self.b = float(right)
        self.n_params = len(armax_result.params) - 1

    def n_parameters(self):
        return self.n_params

    def __call__(self, x):
        return np.sum(stats.truncnorm.logpdf(x, self.a, self.b, self._location,
                self._scale))

    def sample(self, n=1):
        return np.array(stats.truncnorm(self.a, self.b, self._location,
                self._scale).rvs(n))

