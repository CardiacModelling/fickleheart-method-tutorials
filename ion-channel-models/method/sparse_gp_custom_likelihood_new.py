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
        self.sf2 = tt.square(kernelvariance)
    
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

def _create_theano_likelihood_graph_voltage(data, X, ind_X, n_X, n_inducing_X, use_open_prob=False, approx='FITC'):

    rho = tt.dvector('rho')
    ker_sigma = tt.dscalar('ker_sigma')
    sigma = tt.dscalar('sigma')
    V = theano.tensor.as_tensor_variable(X)
    inducing_V = theano.tensor.as_tensor_variable(ind_X)
    y = theano.tensor.as_tensor_variable(data)

    current = tt.dvector('current')
    if use_open_prob:
        rho = tt.dvector('rho')
        open_prob = tt.dvector('open_prob').reshape((-1,1))
        ind_open_prob = tt.dvector('ind_open_prob').reshape((-1,1))
        V_O = tt.concatenate((V, open_prob),axis=1)
        inducing_V_O = tt.concatenate((inducing_V, ind_open_prob),axis=1)
        
    else:
        rho = tt.dscalar('rho')
        V_O = V
        inducing_V_O = inducing_V

    cov_func = RbfKernel(rho, ker_sigma)

    sigma2 = tt.square(sigma)
    Kuu = cov_func(inducing_V_O)
    Kuf = cov_func(inducing_V_O, V_O)
 
    Luu = cholesky(stabilize(Kuu))
    A = solve_lower(Luu, Kuf)
    Qffd = tt.sum(A * A, 0)
    
    if approx == 'FITC':
        Kffd = cov_func(V_O, diag=True)
        Lamd = tt.clip(Kffd - Qffd, 0.0, np.inf) + sigma2
        trace = 0.0
    elif approx == 'VFE':
        Lamd = tt.ones_like(Qffd) * sigma2
        trace = ((1.0 / (2.0 * sigma2)) *
                    (tt.sum(cov_func(V_O, diag=True)) -
                    tt.sum(tt.sum(A * A, 0))))
    else:  # DTC
        Lamd = tt.ones_like(Qffd) * sigma2
        trace = 0.0
    
    A_l = A / Lamd
    L_B = cholesky(tt.eye(n_inducing_X) + tt.dot(A_l, tt.transpose(A)))
    r = y - current
    
    r_l = r / Lamd
    
    c = solve_lower(L_B, tt.dot(A, r_l))
    
    constant = 0.5 * n_X * tt.log(2.0 * np.pi)
    logdet = 0.5 * tt.sum(tt.log(Lamd)) + tt.sum(tt.log(tt.diag(L_B)))
    quadratic = 0.5 * (tt.dot(r, r_l) - tt.dot(c, c))
    ll = -1.0 * (constant + logdet + quadratic + trace)  

    if use_open_prob:
        return theano.function([current,open_prob,ind_open_prob,rho,ker_sigma,sigma],ll,on_unused_input='ignore')
    else:
        return theano.function([current,rho,ker_sigma,sigma],ll,on_unused_input='ignore')


def _create_theano_conditional_graph_voltage(data, X, ind_X, X_new, use_open_prob=False, approx='FITC'):

        
        ker_sigma = tt.dscalar('ker_sigma')
        sigma = tt.dscalar('sigma')
        V = theano.tensor.as_tensor_variable(X)
        V_new = theano.tensor.as_tensor_variable(X_new)
        inducing_V = theano.tensor.as_tensor_variable(ind_X)
        y = theano.tensor.as_tensor_variable(data)
        current = tt.dvector('current')
        current_new = tt.dvector('current_new')

        if use_open_prob:
            rho = tt.dvector('rho')
            open_prob = tt.dvector('open_prob').reshape((-1,1))
            ind_open_prob = tt.dvector('ind_open_prob').reshape((-1,1))
            open_prob_new = tt.dvector('open_prob_new').reshape((-1,1))
            V_O = tt.concatenate((V, open_prob),axis=1)
            inducing_V_O = tt.concatenate((inducing_V, ind_open_prob),axis=1)
            V_O_new = tt.concatenate((V_new, open_prob_new),axis=1)
            
        else:
            rho = tt.dscalar('rho')
            V_O = V
            inducing_V_O = inducing_V   
            V_O_new = V_new   
              

        cov_func = RbfKernel(rho, ker_sigma)

        sigma2 = tt.square(sigma)
        Kuu = cov_func(inducing_V_O)
        Kuf = cov_func(inducing_V_O, V_O)
        Luu = cholesky(stabilize(Kuu))
        A = solve_lower(Luu, Kuf)
        Qffd = tt.sum(A * A, 0)
        if approx == "FITC":
            Kffd = cov_func(V_O, diag=True)
            Lamd = tt.clip(Kffd - Qffd, 0.0, np.inf) + sigma2
        else:  # VFE or DTC
            Lamd = tt.ones_like(Qffd) * sigma2
        A_l = A / Lamd
        L_B = cholesky(tt.eye(inducing_V_O.shape[0]) + tt.dot(A_l, tt.transpose(A)))
        r = y - current
        r_l = r / Lamd
        c = solve_lower(L_B, tt.dot(A, r_l))
        Kus = cov_func(inducing_V_O, V_O_new)
        As = solve_lower(Luu, Kus)
        mu = current_new + tt.dot(tt.transpose(As), solve_upper(tt.transpose(L_B), c))#
        C = solve_lower(L_B, As)
        Kss = cov_func(V_O_new, diag=True)
        var = Kss - tt.sum(tt.square(As), 0) + tt.sum(tt.square(C), 0)
        var += sigma2
        if use_open_prob:

            return [theano.function([current,current_new,open_prob,ind_open_prob,open_prob_new,rho,ker_sigma,sigma],mu,on_unused_input='ignore'), \
                    theano.function([current,current_new,open_prob,ind_open_prob,open_prob_new,rho,ker_sigma,sigma],var,on_unused_input='ignore')]                       
        else:
            return [theano.function([current,current_new,rho,ker_sigma,sigma],mu,on_unused_input='ignore'), \
                    theano.function([current,current_new,rho,ker_sigma,sigma],var,on_unused_input='ignore')]        
        
class DiscrepancyLogLikelihood(pints.ProblemLogLikelihood):
    """
    This class defines a custom loglikelihood which implements a
    discrepancy model where the discrepancy is modelled 
    as a reduced rank Gaussian process using the FITC likelihood
    """

    def __init__(self, problem, voltage, num_ind_thin, use_open_prob=False, downsample=None):
        super(DiscrepancyLogLikelihood, self).__init__(problem)

        self._no = problem.n_outputs()
        self._np = problem.n_parameters()
        self._nt = problem.n_times() 
        self._num_ind_thin = num_ind_thin
        self._voltage = voltage[:,None]
        self._inducing_voltage = self._voltage[::num_ind_thin,:]
        self._nu = len(self._inducing_voltage)
        self._downsample = downsample if downsample is not None else 1
        self._use_open_prob = use_open_prob
        data = self._values[::self._downsample].reshape((-1,))
        v = self._voltage[::self._downsample].reshape((-1,1)) 
        ind_v = self._inducing_voltage.reshape((-1,1))
        
        if self._use_open_prob:
            self._nds = 4
        else:
            self._nds = 3

        self._loglikelihood = _create_theano_likelihood_graph_voltage(data, v, ind_v, self._nt, self._nu, use_open_prob=self._use_open_prob)

        self._n_parameters = self._np + self._nds
                

    def __call__(self, x):

        if self._use_open_prob:
            _sigma, _rho1, _rho2, _ker_sigma = x[-self._nds:]
            _rho = np.append(_rho1, _rho2)
        else:
            _sigma, _rho, _ker_sigma = x[-self._nds:]

        Utx_sigma, Utx_rho, Utx_ker_sigma = np.exp(_sigma), np.exp(_rho), np.exp(_ker_sigma)

        model_params = x[:-self._nds]

        if self._use_open_prob:
            # sim_current, open_prob = self._problem.evaluate(model_params)
            sim_current, open_prob = self._problem._model.simulate(model_params, self._times)
            sim_current = sim_current[::self._downsample].reshape((-1,)).astype(np.float32)
            open_prob = open_prob[::self._downsample].reshape((-1,)).astype(np.float32)
            ind_open_prob = open_prob[::self._num_ind_thin,:]
            return self._loglikelihood(sim_current,open_prob,ind_open_prob,Utx_rho,Utx_ker_sigma,Utx_sigma)
        else:
            sim_current = self._problem.evaluate(model_params)[::self._downsample].reshape((-1,)).astype(np.float32)
            return self._loglikelihood(sim_current,Utx_rho,Utx_ker_sigma,Utx_sigma)

    

