import pints
import numpy as np
import theano
import theano.tensor.slinalg
import theano.tensor as tt
THEANO_FLAGS='optimizer=fast_compile'
theano.config.floatX = 'float32'
theano.config.exception_verbosity= 'high'

cholesky = tt.slinalg.cholesky
solve_lower = tt.slinalg.Solve(A_structure='lower_triangular')
solve_upper = tt.slinalg.Solve(A_structure='upper_triangular')
solve = tt.slinalg.Solve(A_structure='general')

def stabilize(K):
    """ adds small diagonal to a covariance matrix """
    return K + 1e-6 * tt.identity_like(K)

class GpCovariance(object):
    
    def __init__(self, lengthscale, kernelvariance):
        self.ls = lengthscale
        self.sf2 = kernelvariance
    
    def square_dist(self, X, Xs):
        X = tt.mul(X, 1.0 / self.ls)
        X2 = tt.sum(tt.square(X), 1)
        if Xs is None:
            sqd = -2.0 * tt.dot(X, tt.transpose(X)) + (
                tt.reshape(X2, (-1, 1)) + tt.reshape(X2, (1, -1))
            )
        else:
            Xs = tt.mul(Xs, 1.0 / self.ls)
            Xs2 = tt.sum(tt.square(Xs), 1)
            sqd = -2.0 * tt.dot(X, tt.transpose(Xs)) + (
                tt.reshape(X2, (-1, 1)) + tt.reshape(Xs2, (1, -1))
            )
        return tt.clip(sqd, 0.0, np.inf)

class RbfKernel(GpCovariance):

    def __call__(self, X, Xs=None, diag=False):

        if diag==True:
            return self.sf2*tt.alloc(1.0, X.shape[0])
        else:
            return self.sf2*tt.exp(-0.5 * self.square_dist(X, Xs))

def _create_theano_likelihood_graph(data, t, ind_t, n_time, n_inducing_time, approx='FITC'):
    """ 
    Here we use theano to compile a comutational graph defining our discrepancy
    likelihood. Note it just compiles this graph as a C program which will
    get successively called in pints. 
    Thus all the variables defined here are simply placeholders.
    """
    rho = tt.dscalar('rho')
    ker_sigma = tt.dscalar('ker_sigma')
    sigma = tt.dscalar('sigma')
    time = theano.tensor.as_tensor_variable(t)#tt.dmatrix('time')
    inducing_time = theano.tensor.as_tensor_variable(ind_t)#tt.dmatrix('inducing_time')
    y = theano.tensor.as_tensor_variable(data)#tt.dvector('y')
    current = tt.dvector('current')

    cov_func = RbfKernel(rho, ker_sigma)

    sigma2 = tt.square(sigma)
    Kuu = cov_func(inducing_time)
    Kuf = cov_func(inducing_time, time)
 
    Luu = cholesky(stabilize(Kuu))
    A = solve_lower(Luu, Kuf)
    Qffd = tt.sum(A * A, 0)
    
    if approx == 'FITC':
        Kffd = cov_func(time, diag=True)
        Lamd = tt.clip(Kffd - Qffd, 0.0, np.inf) + sigma2
        trace = 0.0
    elif approx == 'VFE':
        Lamd = tt.ones_like(Qffd) * sigma2
        trace = ((1.0 / (2.0 * sigma2)) *
                    (tt.sum(cov_func(time, diag=True)) -
                    tt.sum(tt.sum(A * A, 0))))
    else:  # DTC
        Lamd = tt.ones_like(Qffd) * sigma2
        trace = 0.0
    
    A_l = A / Lamd
    L_B = cholesky(tt.eye(n_inducing_time) + tt.dot(A_l, tt.transpose(A)))
    r = y - current
    
    r_l = r / Lamd
    
    c = solve_lower(L_B, tt.dot(A, r_l))
    
    constant = 0.5 * n_time * tt.log(2.0 * np.pi)
    logdet = 0.5 * tt.sum(tt.log(Lamd)) + tt.sum(tt.log(tt.diag(L_B)))
    quadratic = 0.5 * (tt.dot(r, r_l) - tt.dot(c, c))
    ll = -1.0 * (constant + logdet + quadratic + trace)  
    return theano.function([current,rho,ker_sigma,sigma],ll,on_unused_input='ignore')

class DiscrepancyLogLikelihood(pints.ProblemLogLikelihood):
    """
    This class defines a custom loglikelihood which implements a
    discrepancy model where the discrepancy is modelled 
    as a reduced rank Gaussian process using the FITC likelihood
    """

    def __init__(self, problem, inducing_times, downsample=None, temperature=None):
        super(DiscrepancyLogLikelihood, self).__init__(problem)

        self._no = problem.n_outputs()
        self._np = problem.n_parameters()
        self._nt = problem.n_times() 
        self._inducing_times = inducing_times
        self._nu = len(inducing_times)
        self._downsample = downsample if downsample is not None else 1

        self._nds = 3
        data = self._values[::self._downsample].reshape((-1,))
        t = self._times[::self._downsample].reshape((-1,1)) 
        ind_t = self._inducing_times.reshape((-1,1))
        self._loglikelihood = _create_theano_likelihood_graph(data, t, ind_t, self._nt, self._nu)
        self._n_parameters = self._np + self._nds
        
        self._temperature = temperature
        

    def __call__(self, x):
 
        _sigma, _rho, _ker_sigma = x[-self._nds:]
        Utx_sigma, Utx_rho, Utx_ker_sigma = np.exp(_sigma), np.exp(_rho), np.exp(_ker_sigma)
        model_params = x[:-self._nds]
        sim_current = self._problem.evaluate(model_params)[::self._downsample].reshape((-1,)).astype(np.float32)
        

        if self._temperature==None:
            return self._loglikelihood(sim_current,Utx_rho,Utx_ker_sigma,Utx_sigma)
        else:
            return self._temperature*self._loglikelihood(sim_current,Utx_rho,Utx_ker_sigma,Utx_sigma)
    
