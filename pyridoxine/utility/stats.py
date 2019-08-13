""" Provide statistical functions """

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.optimize as spopt
import scipy.stats as spstats
import scipy.linalg as spla
from .. import plt as rxplt
import corner
import emcee
from multiprocess import Pool
from functools import partial
import warnings


class StatsTraits:
    """ Get stats traits of some data """

    def __init__(self, arr, axis=None):
        """
        Grab data and calculate statistical quantities
        :param arr: data with any dimensions
        :param axis: which axis to perform calculations in the first place
        """

        if arr is None:
            raise ValueError("No data to perform analyses")

        try:
            self.data = np.asarray(arr)
            if arr.size == 0:
                raise ValueError("No data to perform analyses")
        except Exception as e:
            print(e)

        self.axis = axis

        self.mean = self.__apply_func(np.mean)
        self.median = self.__apply_func(np.median)
        self.min = self.__apply_func(np.min)
        self.max = self.__apply_func(np.max)
        if self.data.dtype == int:
            self.mode = self.__apply_func(spstats.mode)

        self.std = self.__apply_func(np.std)

    def __apply_func(self, f):
        return f(self.data, axis=self.axis)


def _make_corner_plot(dim, samples, bins=50, peaks=None, labels=None):
    """
    Plot all the 1D & 2D projections of the posterior probability distributions of parameters
    :param dim: # of the dimension of the parameters space (how many parameters)
    :param samples: all samples from walkers after burn-in steps
    :param bins: # of bins to draw the 1D histogram
    :param peaks: # peaks from MCMC fit
    :param labels: labels for parameters
    :return: figure object and axes object
    """

    fig = corner.corner(samples, bins=50, quantiles=[0.16, 0.5, 0.84], labels=labels)
    axes = np.array(fig.axes).reshape((dim, dim))  # Extract the axes
    if peaks is not None:
        for i in range(dim):  # Loop over the diagonal
            axes[i, i].axvline(peaks[i], ls="-.", color="r")
    return fig, axes


def _make_chain_plot(dim, sampler, labels=None):
    """
    Draw the time series of the parameters from each walker at each step
    :param dim: # of parameters
    :param sampler: all samples from walkers
    :param labels: labels of parameters
    :return: figure object and axes object
    """

    fig, ax = plt.subplots(dim, 1)
    for i in range(dim):
        ax[i].plot(sampler.chain[:, :, i].T)
        if labels is not None: ax[i].set_ylabel(labels[i])
    rxplt.cut_space(fig, ax)
    return fig, ax


def do_mcmc(dim, num_walkers, num_steps, burn_in, guess, ln_prob, x,
            relative_guess=False, guess_spread=1e-4, vectorize=False, pool=None, progress=True,
            bins=50, silent=False, float_format="{:.4f}",
            draw_chains=False, draw_corner=False, labels=None, return_sampler=False):
    """
    Perform MCMC with customized options
    :param dim: # of parameters (in fact dimension of the parameter space)
    :param num_walkers: # of walkers employed
    :param num_steps: # of steps to explore
    :param burn_in: # of steps needed for the chains to “forget” where it started
    :param guess: # initial guess for all the walkers to start with
    :param ln_prob: # the full log-likelihood function in the form of ln_prob(theta, x)
    :param x: data needed to perform likelihood calculation (can be 2d arrays, i.e., [x, y])
    :param relative_guess: initialize walkers in a Gaussian ball around guess (relatively or absolutely)
    :param guess_spread: how spread the Gaussian ball should be
    :param vectorize: if ln_prob accept vectorized parameters (theta), toggle it for better performance
    :param pool: multiprocessing thread pool or MPI pool for emcee to use
    :param progress: whether to show the progress bar
    :param bins: # of bins to calculate peak positions and plotting
    :param silent: # whether to output log
    :param float_format: # format for float number in log
    :param draw_chains: whether to draw chain plot
    :param draw_corner: whether to draw corner plot
    :param labels: axes labels for chain plot and corner plot
    :param return_sampler: whether to return the sampler object
    :return: results or results + sampler object

    For fast sampling (ln_prob runs very fast), pool can speed up if num_walkers is large (>100).
    However, pool may impact the speed if num_walkers is not large but num_steps is large (>1000).
    For slow sampling, I haven't got any test problems.
    Using lambda to map ln_prob to pos does not help at all (map is already used inside emcee if not vectorize).
    """
    
    tmp = guess if relative_guess else 1
    pos = np.array([guess + tmp * guess_spread * np.random.randn(dim) for i in range(num_walkers)])
    if vectorize:
        sampler = emcee.EnsembleSampler(num_walkers, dim, ln_prob, args=(x,), vectorize=True)
    else:
        sampler = emcee.EnsembleSampler(num_walkers, dim, ln_prob, args=(x,), pool=pool)  # pool is None by default

    sampler.reset()  # sampler.clear_chain() for emcee < 3.0
    sampler.run_mcmc(pos, num_steps, progress=progress)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    # sampler.chain[:, burn_in:, :].reshape((-1, dim)) for emcee < 3.0

    st = StatsTraits(samples, axis=0)  # median is designed to be the answer (with max likelihood)
    peaks = np.zeros(dim)
    for i in range(dim):
        hist, bin_edges = np.histogram(samples[:, i], bins, density=True)
        peaks[i] = (bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist) + 1]) / 2.
    sln = np.array([st.median, st.mean, peaks])
    sln_ln_prob = ln_prob(sln, x) if vectorize else np.array([ln_prob(t, x) for t in sln])

    np.set_printoptions(formatter={'float': float_format.format})
    if not silent:
        print("median footsteps:", sln[0], "(std:", st.std, "), mean:", sln[1], ", mode(peak):", sln[2])
        print("median/mean/mode log(likelihood) = ", sln_ln_prob, "; best solution: ", sln[sln_ln_prob.argmax()])
    try:
        ac_time = sampler.get_autocorr_time()
        if not silent:  # only for emcee > 3.0
            print("estimated integrated auto-correlation time (afterward chains are sufficiently converged)", ac_time)
    except Exception as e:
        print(str(e))
    np.set_printoptions()  # reset numpy print options

    if draw_chains: _make_chain_plot(dim, sampler, labels=labels)  # use sampler for the entire chain
    if draw_corner: _make_corner_plot(dim, samples, labels=labels)

    return (sln, sampler) if return_sampler else sln[sln_ln_prob.argmax()]


class UniVarDistribution:
    """ a basic univariate distribution class with fitting procedures

        To construct this class, you need
        :param m: data (following the notation of mass distribution)
        :param t_guess: initial guess for the distribution parameters (vector theta)

        This base class includes the example implementation of a variably tapered power law
        distribution in the cumulative distribution function, probability density function,
        and the corresponding Jacobian matrix function and Hessian matrix function:

        P(>M) = (M/M_min)^alpha * Exp[- (M^beta - M_min^beta) / M_exp^beta],

        let x = ln(M/M_min), the formula above can be rewritten as:
        P(>x) = Exp[-alpha x - Exp(-beta x_exp) (Exp(beta x) - 1)],
        p(x)  = Exp[-alpha x - Exp(-beta x_exp) (Exp(beta x) - 1)]
                    * (alpha + beta * Exp[beta * (x - x_exp)]),
        where the implicit assumptions are
        [-] beta > 0
        [-] alpha != 0
        [-] 0 <= x_exp <= x_max (max from data).

        To construct a new distribution model, you may want to overwrite:
        [-] self.cumulative_func
        [-] self.prob_den_func
        [-] self.jac_func (optional, but needed by self.minus_ln_prob_min if using L-BFGS-B or TNC)
        [-] self.hess_func (optional)
        Also, don't forget to apply appropriate self.bounds.

        All the Jacobian vectors and Hessian matrices of those distribution classes
        have been examined and tested with Mathematica + Python.
    """

    def __init__(self, m, t_guess):
        """
        Construct a class to describe a basic univariate distribution
        :param m: data (following the notation of mass distribution)
        :param t_guess: initial guess for the distribution parameters (vector theta)
        """

        self.m = np.sort(m)  # m is real data, sort it first
        self.m0 = m[0]       # minimum point, as a must for fitting

        # let x = ln(M/M_min) ==> change data range to [0, Infinity]
        # N.B., p(x) = p(m) * dm/dx; m/x is assumed to extend to infinity
        self.x = np.log(self.m / self.m0)
        self.x_CDF = np.arange(self.x.size, 0, -1, dtype=int)/self.x.size  # data CDF

        self.t = np.array(t_guess).T  # e.g., t = [alpha, beta]
        # distribution parameters are transposed to be a vertical vector

        # There are two ways to override the PDF, CDF or related quantities:
        # - with a lambda function or external/static function
        # - define a new member method to utilize other members

        """ I initially defined them as function object members. However, child classes somehow are not able
            to override them. Thus I moved them to be simple class methods that can be overridden easily.
        self.cumulative_func = lambda x, t: np.squeeze(np.exp(-t[0]*x - np.exp(-t[1]*t[2]) * (np.exp(t[1]*x) - 1)))
        self.prob_den_func = lambda x, t: np.squeeze(
            np.exp(-t[0]*x - np.exp(-t[1]*t[2]) * (np.exp(t[1]*x) - 1)) * (t[0] + t[1] * np.exp(t[1]*(x-t[2]))))
        """

        # bounds = ndarray of [min, max] pairs for each parameter for evaluating ln_prob
        self.bounds = np.array([[-np.inf, np.inf], [0., np.inf], [0., self.x[-1]]])

        # For MCMC likelihood exploration
        self.mcmc_dim = self.t.size
        self.mcmc_num_walkers = 32
        self.mcmc_num_steps = 10000
        self.mcmc_burn_in = 1000
        self.sln_mcmc = self.t

        # For minus likelihood minimization with scipy
        self.mini_method = 'Nelder-Mead'
        self.mini_fallback_method = 'Powell'
        self.mini_tol = 1e-12
        self.mini_options = None
        self.mini_fallback_options = None
        self.mini_fallback_or_not = True
        self.__mini_default_options = {}

        # For bootstrapping (short for bs)
        self.bs_samples = None
        self.bs_t = None
        self.bs_t_std = None
        # median likelihood of -ln_prob(t_bs, x_bs), i.e., all bs samples given their own t_bs (fitting required)
        self.bs_likelihood = 0
        # median likelihood of -ln_prob(t_bf, x_bs), i.e., all bs samples given the best-fit t of the original data
        self.bs_likelihood_given_t = None

    def _mini_default_options(self, method_name):
        """ Return the default minimization option by method name
            This function provides on-call default options b/c the tolerance is variable
        """

        self.__mini_default_options = {
            'Nelder-Mead': {'xatol': self.mini_tol, 'fatol': self.mini_tol, 'maxfev': int(1e6)},
            'Powell': {'xtol': self.mini_tol, 'ftol': self.mini_tol, 'maxfev': int(1e6)},
            'BFGS': {'gtol': self.mini_tol, 'maxiter': int(2e5)},
            'CG': {'gtol': self.mini_tol, 'maxiter': int(2e5)},
            'L-BFGS-B': {'ftol': self.mini_tol, 'gtol': self.mini_tol, 'maxfun': int(2e5)},
            'TNC': {'ftol': self.mini_tol, 'gtol': self.mini_tol, 'xtol': self.mini_tol, 'maxiter': int(2e5)},
            'Newton-CG': {'xtol': self.mini_tol, 'maxiter': int(2e5)}
        }
        return self.__mini_default_options[method_name]

    def cumulative_func(self, x, t):
        """ Calculate the cumulative distribution value at x (data) given t (parameters) """

        return np.exp(-t[0]*x - np.exp(-t[1]*t[2]) * (np.exp(t[1]*x) - 1))

    def cdf(self, x, t=None):
        """
        Return the cumulative distribution function evaluation defined by self.cumulative_func
        :param x: non-negative random variable in this distribution
        :param t: distribution parameters (vector theta)
        """

        tmp_t = [np.atleast_2d(_t).T for _t in (self.t if t is None else np.asarray(t).T)]
        return np.squeeze(self.cumulative_func(x, tmp_t))

    def prob_den_func(self, x, t):
        """ Calculate the probability density value at x (data) given t (parameters) """

        return np.exp(-t[0]*x - np.exp(-t[1]*t[2]) * (np.exp(t[1]*x) - 1)) * (t[0] + t[1] * np.exp(t[1]*(x-t[2])))

    def pdf(self, x, t=None):
        """
        Return the probability density function evaluation defined by self.prob_den_func
        :param x: non-negative random variable in this distribution
        :param t: distribution parameters (vector theta)
        """

        tmp_t = [np.atleast_2d(_t).T for _t in (self.t if t is None else np.asarray(t).T)]
        return np.squeeze(self.prob_den_func(x, tmp_t))

    def jac_func(self, x, t):
        """ Return the Jacobian matrix for the example distribution
            N.B.,
                1), the summation over x is taken out to enable this function to accept
                a theta vector list;
                2), for gradient item without x, np.zeros_like(t[0] + x) should be added to
                obtain the real summation results (since the item goes to each (dL/dt)_x
        """

        a, b, x_exp = t  # = [alpha, beta, x_exp] in the variably tapered power law example
        fac1 = x - x_exp
        fac2 = np.exp(b * fac1)
        fac3 = np.exp(-b * x_exp)
        de = a + b * fac2  # de means denominator
        return np.array([1 / de - x,
                         (b * (-x_exp) * fac3 * fac2 - a * fac3 * (x_exp + np.exp(b * x) * fac1)
                                 + fac2 * (1 - b * (fac2 - 1) * fac1)) / de,
                         -(b * fac3 * (1 + np.exp(b * x) * (b / de - 1)))]).sum(axis=-1).T

    def grad_likelihood(self, x, t=None):
        """
        Return the Jacobian matrix evaluation defined by self.jac_func
        :param x: non-negative random variable in this distribution
        :param t: distribution parameters (vector theta)
        """

        tmp_t = [np.atleast_2d(_t).T for _t in (self.t if t is None else np.asarray(t).T)]
        return np.squeeze(self.jac_func(x, tmp_t))

    def hess_func(self, x, t):
        """ Return the Hessian matrix for the example distribution
            N.B.,
                1), the summation over x is taken out to enable this function to accept
                a theta vector list;
                2), for Hessian item without x, np.zeros_like(t[0] + x) should be added to obtain
                the real summation results (since the item goes to each (d^2L/dt^2)_x
                3), the final results is 3 dimensional ndarray, with the last axis = the
                number of theta vectors in the input list, so np.moveaxis is needed
        """

        a, b, x_exp = t  # = [alpha, beta, x_exp] in the variably tapered power law example
        fac1 = b * (x - x_exp)
        fac2 = np.exp(fac1)
        fac3 = 1 + fac1
        fac4 = x - x_exp
        fac5 = np.exp(-b * x_exp)
        de = (a + b * fac2) ** 2  # contains x
        norm22 = a ** 2 * fac5 * (x_exp ** 2 - np.exp(b * x) * fac4 ** 2) \
                 + a * fac2 * (2 * b * x_exp ** 2 * fac5 + fac4 * (2 + b * fac4)) \
                 + fac2 ** 2 * (-1 - 2 * a * b * fac4 ** 2 + b ** 2 * fac5 * (x_exp ** 2 - np.exp(b * x) * fac4 ** 2))
        norm23 = fac5 * (-1 + np.exp(b * x) * (1 + b * fac4) + b * x_exp) \
                 - b * fac2 * (b * fac2 + a * (2 + b * fac4)) / de
        norm33 = b ** 2 * (2 * a * b * fac2 ** 2 - a * b * fac2 * (1 + 2 * fac5) + fac5 * (np.exp(b * x) - 1)
                           * (a ** 2 + b ** 2 * fac2 ** 2))
        return np.moveaxis(np.array([
            [-1 / de, -fac2 * fac3 / de, b ** 2 * fac2 / de],
            [-fac2 * fac3 / de, norm22 / de, norm23],
            [b ** 2 * fac2 / de, norm23, -norm33 / de]
        ]).sum(axis=-1), -1, 0)

    def hess_likelihood(self, x, t=None):
        """
        Return the Hessian matrix evaluation defined by self.hess_func
        :param x: non-negative random variable in this distribution
        :param t: distribution parameters (vector theta)
        """

        tmp_t = [np.atleast_2d(_t).T for _t in (self.t if t is None else np.asarray(t).T)]
        return np.squeeze(self.hess_func(x, tmp_t))

    def _fitting_bounds(self, x):
        """
        Return boundaries for fitting based on input x (mainly for fitting bootstrap samples, where x is variable)
        :param x: non-negative random variable in this distribution
        """

        bounds = np.copy(self.bounds)
        bounds[2][1] = x[-1]
        return bounds

    def ln_prob(self, t, x):
        """
        Calculate the log-likelihood of data (x) given the distribution parameters (t)
        :param x: non-negative random variable in this distribution
        :param t: distribution parameters (vector theta); can accept an array of vector theta
        """

        t = np.asarray(t)
        L = np.atleast_1d(np.sum(np.log(self.pdf(x, t=t)), axis=t.ndim - 1))
        L[np.isnan(L)] = -np.inf
        # the customized bounds serve for the purpose of ln_prior
        bounds = self._fitting_bounds(x)
        if bounds is not None:
            for i, item in enumerate([np.atleast_1d(_t) for _t in t.T]):
                L[(item<bounds[i][0])|(item>bounds[i][1])] = -np.inf
        return np.squeeze(L)

    def ln_prob_wo_prior(self, t, x):
        """
        Calculate the log-likelihood of data (x) given the distribution parameters (t) without prior check
        :param x: non-negative random variable in this distribution
        :param t: distribution parameters (vector theta); can accept an array of vector theta
        """

        t = np.asarray(t)
        L = np.atleast_1d(np.sum(np.log(self.pdf(x, t=t)), axis=t.ndim - 1))
        L[np.isnan(L)] = -np.inf

        return np.squeeze(L)

    def mcmc_fitting(self, x, t_guess=None, **kwargs):
        """
        Do MCMC fitting to explore the parameter space
        :param x: non-negative random variable in this distribution
        :param t_guess: initial guess for the distribution parameters (vector theta)
        :param kwargs: other keywords for do_mcmc
        """
        if self._fitting_bounds(x) is None:
            Warning("No boundary has been set for parameters.")
        return do_mcmc(self.mcmc_dim, self.mcmc_num_walkers, self.mcmc_num_steps, self.mcmc_burn_in,
                       (self.t if t_guess is None else np.asarray(t_guess)),
                       self.ln_prob, x, vectorize=True, **kwargs)

    def test_mini_method(self, t_guess, x, tol=range(8, 9), max_f_i=int(2e5), more_methods=False, methods=None,
                         ln_prob_format=r'{:.2f}', t_format=r'{:+.2e}', no_warnings=True, disp=False):
        """
        Test various minimization method with scipy.optimize.minimize
        :param t_guess: initial guess for the distribution parameters (vector theta)
        :param x: non-negative random variable in this distribution
        :param tol: array, list or tuple, all kinds of tolerance for successful termination
        :param max_f_i: scalar, maximum number of function evaluation / iteration
        :param more_methods: bool, whether or not to include more methods in testing
        :param methods: a list (or tuple) of strings, customized methods by users
        :param ln_prob_format: formatter for -ln-L
        :param t_format: formatter for parameters
        :param no_warnings: disable warnings
        :param disp: whether to add disp into minimization options
        """

        if methods is None:
            all_methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC']
            if more_methods:
                all_methods += ['dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact']
        else:
            all_methods = methods
        bounds = self._fitting_bounds(x)

        with warnings.catch_warnings():
            if no_warnings: warnings.simplefilter('ignore')
            if tol is None:
                if max_f_i is None:
                    print("Use default minimization options")
                else:
                    print("Use default tolerance with maxfev/maxfun/maxiter=", max_f_i)
                for meth in all_methods:
                    if max_f_i is None:
                        res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess, method=meth, bounds=bounds,
                                             jac=lambda t: -self.grad_likelihood(x, t=t),
                                             hess=lambda t: -self.hess_likelihood(x, t=t), options={'disp': disp})
                    else:
                        res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess, method=meth, bounds=bounds,
                                             jac=lambda t: -self.grad_likelihood(x, t=t),
                                             hess=lambda t: -self.hess_likelihood(x, t=t),
                                             options={'maxfev': max_f_i, 'maxfun': max_f_i,
                                                      'maxiter': max_f_i, 'disp': disp})
                    print((r'method: {:<11} success? {:1} -lnL= '+ln_prob_format+r', grad-L= ['
                          +r'{:+.2e}, '*(self.t.size-1)+r'{:+.2e}] t = ['+(t_format+r', ')*(self.t.size-1)
                          +t_format+']').format(meth, res.success, res.fun, *tuple(self.grad_likelihood(x, t=res.x)),
                                                *tuple(res.x)), flush=True)
            else:
                for ep in tol:
                    print("Now tolerance = 1e-"+str(ep), flush=True)
                    for meth in all_methods:
                        res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess, method=meth, bounds=bounds,
                                             jac=lambda t: -self.grad_likelihood(x, t=t),
                                             hess=lambda t: -self.hess_likelihood(x, t=t),
                                             options={'maxfev': max_f_i, 'maxfun': max_f_i, 'maxiter': max_f_i,
                                                      'xtol': 10**(-ep), 'gtol': 10**(-ep), 'ftol': 10**(-ep),
                                                      'xatol': 10**(-ep), 'fatol': 10**(-ep), 'disp': disp})
                        print((r'method: {:<11} success? {:1} -lnL= '+ln_prob_format+r', grad-L= ['
                              +r'{:+.2e}, '*(self.t.size-1)+r'{:+.2e}] t = ['+(t_format+r', ')*(self.t.size-1)
                              +t_format+']').format(meth, res.success, res.fun,
                                                    *tuple(self.grad_likelihood(x, t=res.x)), *tuple(res.x)),
                              flush=True)

    def _mesh_grid_mini(self, method, x, t_guess, tol=12, max_f_i=int(2e5)):
        """
        sub-method for mesh grid minimization in parameter space
        :param method: which method for scipy.optimize.minimize
        :param x: non-negative random variable in this distribution
        :param t_guess: initial guess for the distribution parameters (vector theta)
        :param tol: array, list or tuple, all kinds of tolerance for successful termination
        :param max_f_i: scalar, maximum number of function evaluation / iteration
        """

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess, method=method, bounds=self._fitting_bounds(x),
                                 jac=lambda t: -self.grad_likelihood(x, t=t),
                                 hess=lambda t: -self.hess_likelihood(x, t=t),
                                 options={'maxfev': max_f_i, 'maxfun': max_f_i, 'maxiter': max_f_i,
                                          'xtol': 10**(-tol), 'gtol': 10**(-tol), 'ftol': 10**(-tol),
                                          'xatol': 10**(-tol), 'fatol': 10**(-tol)})
            return np.hstack([res.fun, res.x])

    def mesh_grid_mini(self, t_guess, x, method, tol=12, max_f_i=int(2e5), threads=6, mesh_ready=False):
        """
        sub-method for mesh grid minimization in parameter space
        :param method: which method for scipy.optimize.minimize
        :param x: non-negative random variable in this distribution
        :param t_guess: initial guess for the distribution parameters (vector theta)
        :param tol: array, list or tuple, all kinds of tolerance for successful termination
        :param max_f_i: scalar, maximum number of function evaluation / iteration
        :param threads: how many threads in the thread pool
        :param mesh_ready: set this to True if t_guess is already an array of parameter vectors
        """
        if mesh_ready:
            mesh_grid_t = t_guess
        else:
            t_guess = np.asarray(t_guess)
            if t_guess.ndim is 1:  # t_guess only have one parameter
                mesh_grid_t = np.atleast_2d(t_guess).T
            elif t_guess.ndim is 2:  # t_guess have multiple parameters (>1)
                mesh_grid_t = np.vstack([item.flatten() for item in np.meshgrid(*tuple(t_guess))]).T
            else:
                raise ValueError("t_guess should be 1D or 2D; here we got: ", t_guess)

        p = Pool(threads)
        tmp_func = partial(self._mesh_grid_mini, method, x, tol=tol, max_f_i=max_f_i)
        mini_lnL_t = np.array(p.map(tmp_func, mesh_grid_t))
        p.close()
        mini_idx = np.argmin(mini_lnL_t[:, 0])
        return mini_lnL_t[mini_idx, 0], mini_lnL_t[mini_idx, 1:]

    def _minus_ln_prob_min(self, t_guess, x, method, silent=True):
        """
        Minimize the minus log likelihood by scipy.optimize.minimize function
        :param t_guess: initial guess for the distribution parameters (vector theta)
        :param x: non-negative random variable in this distribution
        :param method: minimization method for scipy
        """

        bounds = self._fitting_bounds(x)
        if bounds is not None:
            t_guess = np.maximum(t_guess, bounds[:, 0])
            t_guess = np.minimum(t_guess, bounds[:, 1])
            # This is useful if t_guess is out of bound
            # the fitting won't start with a likelihood of -inf and have nowhere to go

        if method in ['Nelder-Mead', 'Powell']:
            if self.mini_options is None: self.mini_options = self._mini_default_options(method)
            if not silent: self.mini_options['disp'] = True
            res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess, method=method,
                                 options=self.mini_options)
        elif method in ['BFGS', 'CG']:
            if self.mini_options is None: self.mini_options = self._mini_default_options(method)
            if not silent: self.mini_options['disp'] = True
            res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess, method=method,
                                 jac=lambda t: -self.grad_likelihood(x, t=t), options=self.mini_options)
        elif method in ['L-BFGS-B', 'TNC']:
            if self.mini_options is None: self.mini_options = self._mini_default_options(method)
            if not silent: self.mini_options['disp'] = True
            res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess, method=method, bounds=bounds,
                                 jac=lambda t: -self.grad_likelihood(x, t=t), options=self.mini_options)
        elif method in ['Newton-CG']:
            if self.mini_options is None: self.mini_options = self._mini_default_options(method)
            if not silent: self.mini_options['disp'] = True
            res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess, method=method, bounds=bounds,
                                 jac=lambda t: -self.grad_likelihood(x, t=t),
                                 hess=lambda t: -self.hess_likelihood(x, t=t), options=self.mini_options)
        else:
            if self.mini_options is None and (not silent): self.mini_options = {'disp': True}
            res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess, method=method, bounds=bounds,
                                 jac=lambda t: -self.grad_likelihood(x, t=t),
                                 hess=lambda t: -self.hess_likelihood(x, t=t), options=self.mini_options)
        return res

    def minus_ln_prob_min(self, t_guess, x, silent=True):
        """
        Minimize the minus log likelihood by scipy.optimize.minimize function
        :param t_guess: initial guess for the distribution parameters (vector theta)
        :param x: non-negative random variable in this distribution
        :param silent: whether or not to print the result
        :param fallback: whether or not to use the fallback method
        Different method yield different options
        """

        if silent and self.mini_options is not None: self.mini_options.pop('disp', None)

        res = self._minus_ln_prob_min(t_guess, x, self.mini_method, silent=silent)
        tmp_t = res.x
        if not res.success and self.mini_fallback_or_not:
            # swap the options
            self.mini_options, self.mini_fallback_options = self.mini_fallback_options, self.mini_options
            if silent and self.mini_options is not None: self.mini_options.pop('disp', None)
            fallback_res = self._minus_ln_prob_min(t_guess, x, self.mini_fallback_method, silent=silent)
            if not fallback_res.success:
                print(r'Optimization terminated unsuccessfully')
                print(x, res, fallback_res)
            # we choose the max likelihood
            if self.ln_prob(tmp_t, x) < self.ln_prob(fallback_res.x, x):
                tmp_t = fallback_res.x
                res = fallback_res
            # swap back
            self.mini_options, self.mini_fallback_options = self.mini_fallback_options, self.mini_options
        if not silent:
            print("minimization results:\n", res)
        return tmp_t  # this is automatically numpy.ndarray

    def maximum_likelihood_eqn_set(self, t, m):
        """
        Define the analytical equation set to solve t for maximum likelihood (if possible)
        :param t: distribution parameters (vector theta)
        :param m: real data
        """

        return None

    def solve_maximum_likelihood_eqn_set(self, m):
        """
        Solve the analytical maximum likelihood equation set to obtain t
        :param m: real data
        """

        return spopt.fsolve(self.maximum_likelihood_eqn_set, self.t, args=(m,))

    def generate_bootstrap_samples(self, num_samples):
        """
        Generate bootstrap samples from the real data instead of x (N.B.)
        :param num_samples: the number of samples needed
        """

        bs_m = np.sort(np.random.choice(self.m, [num_samples, self.m.size], replace=True), axis=1)
        return np.log(bs_m / bs_m[:, 0, None])

    def bootstrap_model_fitting(self, bs_samples, t_guess, std=False, threads=6):
        """
        Apply the fitting to a large number of bootstrap samples
        :param bs_samples: bootstrap samples
        :param t_guess: initial guess for the distribution parameters (vector theta)
        :param std: whether to return all the fitted theta or return their standard deviation
        :param threads: how many threads in the thread pool
        :return: all the fitted theta vectors (or their std) and the averaged likelihood from all samples
        """

        n_bs = bs_samples.shape[0]

        if self.mini_method == self.mini_fallback_method:
            print("Warning: the chosen method is the same as the fallback method.")
            self.mini_fallback_or_not = False
        p = Pool(threads)
        tmp_func = partial(self.minus_ln_prob_min, t_guess)
        bs_t = np.array(p.map(tmp_func, bs_samples))
        p.close()
        bs_likelihood = self.ln_prob(bs_t, self.x)
        if bs_likelihood[~np.isfinite(bs_likelihood)].size > 0:
            print('RuntimeWarning: {:d} bootstrap samples lead to infinite/NAN likelihood'.format(
                bs_likelihood[~np.isfinite(bs_likelihood)].size))
        self.bs_t, self.bs_t_std, self.bs_likelihood = bs_t, bs_t.std(axis=0), \
                                                       -bs_likelihood[np.isfinite(bs_likelihood)].mean()
        return self.bs_t_std, self.bs_likelihood

    def median_bootstrap_ln_prob(self, bs_samples, t_bf):
        """
        Calculate the median of ln_prob(t, bootstrap_samples) to estimate the goodness-of-fit
        :param bs_samples: bootstrap samples
        :param t_bf: the best-fit parameters
        """

        self.bs_likelihood_given_t = -np.median(
            np.array([self.ln_prob_wo_prior(t_bf, bs_sample) for bs_sample in bs_samples]))
        return self.bs_likelihood_given_t

    def bootstrap_eqn_solving(self, bs_samples, std=False, threads=6):
        """
        Apply the fitting to a large number of bootstrap samples
        :param bs_samples: bootstrap samples
        :param std: whether to return all the fitted theta or return their standard deviation
        :param threads: how many threads in the thread pool
        :return: all the fitted theta vectors (or their std) and the averaged likelihood from all samples
        """

        n_bs = bs_samples.shape[0]
        p = Pool(threads)
        bs_t = np.array(p.map(self.solve_maximum_likelihood_eqn_set, bs_samples))
        p.close()
        if (bs_t.shape[0] < n_bs):
            print("RuntimeWarning: {:d} bootstrap samples didn't return theta".format(n_bs - bs_t.shape[0]))
        bs_likelihood = self.ln_prob(bs_t, self.x)
        if bs_likelihood[~np.isfinite(bs_likelihood)].size > 0:
            print('RuntimeWarning: {:d} bootstrap samples lead to infinite/NAN likelihood'.format(
                bs_likelihood[~np.isfinite(bs_likelihood)].size))
        self.bs_t, self.bs_t_std, self.bs_likelihood = bs_t, bs_t.std(axis=0), \
                                                       -bs_likelihood[np.isfinite(bs_likelihood)].mean()
        return self.bs_t_std, self.bs_likelihood

    def fitting(self, use_solver=False, use_mcmc=True,
                jac=False, inv_hess=False, silent=False, f_format="{:.8e}", **kwargs):
        """
        Perform the MCMC exploration and -ln_prob minimization
        :param use_solver: solve the maximum likelihood equation set instead of MCMC&ln_prob
        :param use_mcmc: explore the parameter space for the maximum likelihood with MCMC
        :param jac: print the Jacobian vector
        :param inv_hess: print the inverse Hessian matrix
        :param silent: do not print the fitting results
        :param f_format: number format in jacobian and hessian
        """

        if use_solver:
            self.t = self.solve_maximum_likelihood_eqn_set(self.m)
            if self.t is None:
                raise RuntimeError("It seems the maximum likelihood equation set has not been defined.")
        else:
            if use_mcmc:
                self.sln_mcmc = self.mcmc_fitting(self.x, silent=False, **kwargs)
            print("Chosen minimization method: ", self.mini_method)
            if self.mini_method == self.mini_fallback_method:
                print("Warning: the chosen method is the same as the fallback method.")
                self.mini_fallback_or_not = False
            else:
                self.mini_fallback_or_not = True
            if use_mcmc:
                self.t = self.minus_ln_prob_min(self.sln_mcmc, self.x, silent=False)
            else:
                self.t = self.minus_ln_prob_min(self.t, self.x, silent=False)
                self.sln_mcmc = self.t

        np.set_printoptions(formatter={'float': f_format.format})
        if not silent:
            print("Fitting results:", self.t, self.ln_prob(self.t, self.x))
        if jac:
            print("grad likelihood at theta:", -self.grad_likelihood(self.x, t=self.t))
        if inv_hess:
            print("inv-Hessian matrix at theta:\n", spla.inv(-self.hess_likelihood(self.x, t=self.t)))
        np.set_printoptions()  # reset it


class SimpleTaperedPowerLaw(UniVarDistribution):
    """ Distribution class for a simple tapered power law.

        This class describes a distribution follows:

        P(>M) = (M/M_min)^alpha * Exp[- (M - M_min) / M_exp],

        let x = ln(M/M_min), the formula above can be rewritten as:
        P(>x) = Exp[-alpha x - Exp(-x_exp) (Exp(x) - 1)],
        p(x)  = Exp[-alpha x - Exp(-x_exp) (Exp(x) - 1)] * (alpha + Exp[x - x_exp]),
        where the implicit assumptions are
        [-] alpha != 0
        [-] 0 <= x_exp <= x_max (max from data).
    """

    def __init__(self, m, t_guess):

        super().__init__(m, t_guess)

        self.bounds = np.array([[-np.inf, np.inf], [0, self.x[-1]]])
        self.cumulative_func = lambda x, t: np.exp(-t[0] * x - np.exp(-t[1]) * (np.exp(x) - 1))
        self.prob_den_func = lambda x, t: np.exp(-t[0] * x - np.exp(-t[1])*(np.exp(x)-1)) * (t[0] + np.exp(x - t[1]))

    def maximum_likelihood_eqn_set(self, t, m):
        """
        Define the analytical equation set to solve t for maximum likelihood (if possible)
        :param t: distribution parameters (vector theta)
        :param m: real data
        """

        a, x_exp = t  # = [alpha, x_exp] in the vector theta
        return [np.sum(1 / (a + np.exp(-x_exp) * m / m[0])) - np.sum(np.log(m / m[0])),
                np.sum(m / (a + np.exp(-x_exp) * m / m[0])) - np.sum(m - m[0])]

    def jac_func(self, x, t):
        """ Return the Jacobian vector of the simply tapered power law distribution """

        a, x_exp = t  # = [alpha, x_exp] in the vector theta
        de = a + np.exp(x - x_exp)  # de means denominator
        return np.array([1 / de - x, -np.exp(-x_exp) * (1 + np.exp(x) * (1 / de - 1))]).sum(axis=-1).T

    def hess_func(self, x, t):
        """ Return the Hessian matrix of the simply tapered power law distribution """

        a, x_exp = t # = [alpha, x_exp] in the vector theta
        fac = x - x_exp
        de = (a + np.exp(fac)) ** 2  # de means denominator
        return np.moveaxis(np.array([[         -1 / de, np.exp(fac) / de],
                                     [np.exp(fac) / de, np.exp(-x_exp) - np.exp(fac) + a * np.exp(fac) / de]
                                     ]).sum(axis=-1), -1, 0)

    def _fitting_bounds(self, x):
        """
        Return boundaries for fitting based on x (mainly for fitting bootstrap samples)
        :param x: non-negative random variable in this distribution
        """

        bounds = np.copy(self.bounds)
        bounds[1][1] = x[-1]
        return bounds


class VariablyTaperedPowerLaw(UniVarDistribution):
    """ Distribution class for a variably tapered power law

        The UniVarDistribution already implements necessary functions for the variably tapered power law:

        P(>M) = (M/M_min)^alpha * Exp[- (M^beta - M_min^beta) / M_exp^beta],

        let x = ln(M/M_min), the formula above can be rewritten as:
        P(>x) = Exp[-alpha x - Exp(-beta x_exp) (Exp(beta x) - 1)],
        p(x)  = Exp[-alpha x - Exp(-beta x_exp) (Exp(beta x) - 1)]
                    * (alpha + beta * Exp[beta * (x - x_exp)]),

        N.B.: the implicit assumptions in this wrapper class are weaken to
        [-] alpha and beta cannot be negative simultaneously
        [-] 0 <= x_exp <= x_max (max from data).

    """

    def __init__(self, m, t_guess):

        super().__init__(m, t_guess)

        self.bounds = np.array([[-np.inf, np.inf], [-np.inf, np.inf], [0, self.x[-1]]])

    def ln_prob(self, t, x):
        """
        Calculate the log-likelihood of data (x) given the distribution parameters (t)
        :param x: non-negative random variable in this distribution
        :param t: distribution parameters (vector theta); can accept an array of vector theta
        """

        t = np.asarray(t)
        L = np.atleast_1d(np.sum(np.log(self.pdf(x, t=t)), axis=t.ndim - 1))
        L[np.isnan(L)] = -np.inf
        # the customized bounds serve for the purpose of ln_prior
        bounds = self._fitting_bounds(x)
        if bounds is not None:
            tmp_t = [np.atleast_1d(_t) for _t in t.T]
            L[(tmp_t[0] < 0) & (tmp_t[1] < 0)] = -np.inf
            L[(tmp_t[2] < bounds[2][0]) | (tmp_t[2] > bounds[2][1])] = -np.inf
        return np.squeeze(L)


class TruncatedPowerLaw(UniVarDistribution):
    """ Distribution class for a truncated power law.

        This class describes a distribution follows:

        p(M) = (a / M) * (M / M_min)^(-a) / (1 - (M_tr / M_min)^(-a)),
        p(M) = 0.0 for M > M_tr

        where a stands for alpha for simplicity.

        let x = ln(M/M_min), the formula above can be rewritten as:
        P(>x) = (1 - Exp(a * (x_tr - x))) / (1 - Exp(a x_tr))
        p(x)  = a Exp(-a x) / (1 - Exp(-a x_tr))
        where the implicit assumptions are
        [-] alpha > 0
        [-] x_tr should be >= ln(M_max/M_min) in data
        [-] p(x) = 0 for x > x_tr

        Though, it can be proved analytically that when x_tr = ln(M_max/M_min) in data,
        the log likelihood function reaches its local maximum.
    """

    def __init__(self, m, t_guess):

        super().__init__(m, t_guess)

        self.bounds = np.array([[0, np.inf], [self.x[-1], np.inf]])
        #self.cumulative_func = lambda x, t: (1 - np.exp(t[0] * (t[1] - x))) / (1 - np.exp(t[0] * t[1]))
        #self.prob_den_func = lambda x, t: (t[0] * np.exp(-t[0] * x)) / (1 - np.exp(-t[0] * t[1]))

    def cumulative_func(self, x, t):
        """ Calculate the cumulative distribution value at x (data) given t (parameters)  """

        a, x_tr = t  # = [alpha, x_tr] in the vector theta
        tmp_cdf = (1 - np.exp(t[0] * (t[1] - x))) / (1 - np.exp(t[0] * t[1]))
        tmp_cdf[x > x_tr] = 0.0  # integrate from right to left and nothing there
        return tmp_cdf

    def prob_den_func(self, x, t):
        """ Calculate the probability density value at x (data) given t (parameters) """

        a, x_tr = t  # = [alpha, x_tr] in the vector theta
        tmp_pdf = (t[0] * np.exp(-t[0] * x)) / (1 - np.exp(-t[0] * t[1]))
        tmp_pdf[x > x_tr] = 0.0  # no probability outside x_tr
        return tmp_pdf

    def jac_func(self, x, t):
        """ Return the Jacobian vector of the truncated power law distribution """

        a, x_tr = t  # = [alpha, x_tr] in the vector theta
        de = 1 - np.exp(a * x_tr)
        tmp_jac = np.array([1/a - x + x_tr / de, np.zeros_like(a + x) + a / de])
        tmp_jac[:, x > x_tr] = 0.0
        return tmp_jac.sum(axis=-1).T

    def hess_func(self, x, t):
        """ Return the Hessian matrix of the truncated power law distribution """

        a, x_tr = t  # = [alpha, x_tr] in the vector theta
        de = (1 - np.exp(a * x_tr))**2
        zeros = np.zeros_like(a + x)
        tmp_hess = np.array([
            [zeros + (- 1/a**2 + x_tr**2 / 4 * 1/np.sinh(a * x_tr / 2)**2),
             zeros + (1 + np.exp(a * x_tr) * (a * x_tr - 1)) / de],
            [zeros + (1 + np.exp(a * x_tr) * (a * x_tr - 1)) / de,
             zeros + a**2 / 4 * 1/np.sinh(a * x_tr / 2)**2]
        ])
        tmp_hess[:, :, x > x_tr] = 0.0
        return np.moveaxis(tmp_hess.sum(axis=-1), -1, 0)

    def _fitting_bounds(self, x):
        """
        Return boundaries for fitting based on x (mainly for fitting bootstrap samples)
        :param x: non-negative random variable in this distribution
        """

        bounds = np.copy(self.bounds)
        bounds[1][0] = x[-1]
        return bounds


class BrokenCumulativePowerLaw(UniVarDistribution):
    """ Distribution class for a broken cumulative power law.

        This class describes a distribution follows:

                ┏  (M / M_min)^(-a1)
        P(>M) = |
                ┗  (M / M_min)^(-a2) / (M_br / M_min)^(a2 - a1)
        where a1, a2 stand for alpha_1, alpha_2 for simplicity.

        let x = ln(M/M_min), the formula above can be rewritten as:

                ┏  Exp(-a1 x)
        P(>x) = |
                ┗  Exp(-a2 x + (a2 - a1) x_br)

                ┏  a1 Exp(-a1 x)
        p(x)  = |
                ┗  a2 Exp(-a2 x + (a2 - a1) x_br)

        where the implicit assumptions are
        [-] no constraints on a1
        [-] a2 > 0
        [-] 0 <= x_br <= x_max (max from data)
    """

    def __init__(self, m, t_guess):

        super().__init__(m, t_guess)

        self.bounds = np.array([[-np.inf, np.inf], [0, np.inf], [0, self.x[-1]]])

    def cumulative_func(self, x, t):
        """ Calculate the cumulative distribution value at x (data) given t (parameters)  """

        a1, a2, x_br = t  # = [alpha_1, alpha_2, x_br] in the theta vector
        part1 = np.exp(-a1*x); part2 = np.exp((a2-a1)*x_br-a2*x)
        # old method:
        #mask_x = np.tile(x, [a1.size, 1])
        #mask1 = np.zeros_like(part1); mask1[mask_x<x_br] = 1; mask2 = np.zeros_like(part2); mask2[mask_x>x_br] = 1
        #return part1 * mask1 + part2 * mask2
        # in fact, (x > x_br) has the same shape as part1
        part1[x >= x_br] = 0.0; part2[x < x_br] = 0.0
        return part1 + part2
    
    def prob_den_func(self, x, t):
        """ Calculate the probability density value at x (data) given t (parameters) """

        a1, a2, x_br = t  # = [alpha_1, alpha_2, x_br] in the theta vector
        part1 = a1*np.exp(-a1*x); part2 = a2*np.exp((a2-a1)*x_br-a2*x)
        # old method:
        #mask_x = np.tile(x, [a1.size, 1])
        #mask1 = np.zeros_like(part1); mask1[mask_x<x_br] = 1; mask2 = np.zeros_like(part2); mask2[mask_x>x_br] = 1
        #return part1 * mask1 + part2 * mask2
        # in fact, (x > x_br) has the same shape as part1
        part1[x >= x_br] = 0.0; part2[x < x_br] = 0.0
        return part1 + part2

    def jac_func(self, x, t):
        """ Return the Jacobian vector of the broken cumulative power law distribution """

        a1, a2, x_br = t  # = [alpha_1, alpha_2, x_br] in the theta vector
        zeros = np.zeros_like(a1 + x)  # zeros is in fact 2D array
        part1 = np.array([1/a1-x, zeros, zeros]); part2 = np.array([zeros-x_br, 1/a2+x_br-x, zeros-a1+a2])
        # N.B., (x > x_br) is two dimensional, because x_br is a vertical vector
        # thus, part1[:, x > x_br] will pin down the correct items.
        # Alternatively, use part1[..., x > x_br] to select unwanted items.
        part1[:, x >= x_br] = 0.0; part2[:, x < x_br] = 0.0  # mask out the out-of-scope regimes
        return (part1 + part2).sum(axis=-1).T

    def hess_func(self, x, t):
        """ Return the Hessian matrix of the broken cumulative power law distribution """

        a1, a2, x_br = t  # = [alpha_1, alpha_2, x_br] in the theta vector
        zeros = np.zeros_like(a1 + x)
        part1 = np.array([
            [zeros-1/a1**2, zeros, zeros],
            [zeros, zeros, zeros],
            [zeros, zeros, zeros]
        ])
        part2 = np.array([
            [zeros, zeros, zeros-1],
            [zeros, zeros-1/a2**2, zeros+1],
            [zeros-1, zeros+1, zeros]
        ])
        # N.B., again, (x > x_br) is two dimensional, since x_br is a vertical vector
        # thus, part1[:, :, x > x_br] will pin down the correct items (part1 is 4D here)
        # Alternatively, use part1[..., x > x_br] to select unwanted items
        part1[:, :, x >= x_br] = 0.0; part2[:, :, x < x_br] = 0.0  # mask out the out-of-scope regimes
        return np.moveaxis((part1 + part2).sum(axis=-1), -1, 0)

    def _fitting_bounds(self, x):
        """
        Return boundaries for fitting based on x (mainly for fitting bootstrap samples)
        :param x: non-negative random variable in this distribution
        """

        bounds = np.copy(self.bounds)
        bounds[2][1] = x[-1]
        return bounds


class BrokenPowerLaw(UniVarDistribution):
    """ Distribution class for a broken power law as the PDF.

        This class describes a distribution follows:

               ┏  C_0 (M / M_min)^(-a1)
        p(M) = |
               ┗  C_0 (M / M_min)^(-a2) / (M_br / M_min)^(a2 - a1)
        where a1, a2 stand for alpha_1, alpha_2 for simplicity, and

        C_0 = [ 1/a1 + (1/a2 - 1/a1) * (M_br / M_min)^(-a1) ]^(-1)

        let x = ln(M/M_min), the formula above can be rewritten as:

                ┏  C_0 ( Exp(-a1 x) / a1 + (1/a2 - 1/a1) * Exp(-a1 x_br) )
        P(>x) = |
                ┗  C_0 Exp(-a2 x + (a2 - a1) x_br) / a2

                ┏  C_0 Exp(-a1 x)
        p(x)  = |
                ┗  C_0 Exp(-a2 x + (a2 - a1) x_br)

        C_0 = [ 1/a1 + (1/a2 - 1/a1) * Exp(-a1 x_br) ]^(-1)
        where the implicit assumptions are
        [-] no constraints on a1
        [-] a2 > 0
        [-] 0 <= x_br <= x_max (max from data)
    """

    def __init__(self, m, t_guess):

        super().__init__(m, t_guess)

        self.bounds = np.array([[-np.inf, np.inf], [0, np.inf], [0, self.x[-1]]])

    def cumulative_func(self, x, t):
        """ Calculate the cumulative distribution value at x (data) given t (parameters)  """

        a1, a2, x_br = t  # = [alpha_1, alpha_2, x_br] in the theta vector
        part1 = np.exp(-a1 * x) / a1 + (1 / a2 - 1 / a1) * np.exp(-a1 * x_br)
        part2 = np.exp((a2 - a1) * x_br - a2 * x) / a2
        #mask_x = np.tile(x, [a1.size, 1])
        #mask1 = np.zeros_like(part1); mask1[mask_x < x_br] = 1
        #mask2 = np.zeros_like(part2); mask2[mask_x > x_br] = 1
        #return (part1 * mask1 + part2 * mask2) / (1 / a1 + (1 / a2 - 1 / a1) * np.exp(-a1 * x_br) / a2)
        part1[x >= x_br] = 0.0; part2[x < x_br] = 0.0
        return (part1 + part2)  / (1 / a1 + (1 / a2 - 1 / a1) * np.exp(-a1 * x_br))

    def prob_den_func(self, x, t):
        """ Calculate the probability density value at x (data) given t (parameters) """

        a1, a2, x_br = t  # = [alpha_1, alpha_2, x_br] in the theta vector
        part1 = np.exp(-a1 * x); part2 = np.exp((a2 - a1) * x_br - a2 * x)
        #mask_x = np.tile(x, [a1.size, 1])
        #mask1 = np.zeros_like(part1); mask1[mask_x < x_br] = 1; mask2 = np.zeros_like(part2); mask2[mask_x > x_br] = 1
        #return (part1 * mask1 + part2 * mask2) / (1 / a1 + (1 / a2 - 1 / a1) * np.exp(-a1 * x_br) / a2)
        part1[x >= x_br] = 0.0; part2[x < x_br] = 0.0
        return (part1 + part2) / (1 / a1 + (1 / a2 - 1 / a1) * np.exp(-a1 * x_br))

    def jac_func(self, x, t):
        """ Return the Jacobian vector of the broken power law distribution """

        a1, a2, x_br = t  # = [alpha_1, alpha_2, x_br] in the theta vector
        zeros = np.zeros_like(a1 + x)  # zeros is in fact 2D array
        de = a1 + a2 * (np.exp(a1 * x_br) - 1)  # denominator
        part1 = np.array([1 / a1 - x + ((a1 - a2) * x_br - 1) / de,
                          zeros + a1 / a2 / de,
                          zeros + a1 * (a1 - a2) / de])

        part2 = np.array([zeros + 1 / a1 - x_br + ((a1 - a2) * x_br - 1) / de,
                          a1 / a2 / de - x + x_br,
                          zeros + a2 - a1  * a2 * np.exp(a1 * x_br) / de])

        part1[..., x >= x_br] = 0.0
        part2[..., x < x_br] = 0.0  # mask out the out-of-scope regimes
        return (part1 + part2).sum(axis=-1).T

    def hess_func(self, x, t):
        """ Return the Hessian matrix of the broken power law distribution """

        a1, a2, x_br = t  # = [alpha_1, alpha_2, x_br] in the theta vector

        zeros = np.zeros_like(a1 + x)  # zeros is in fact 2D array
        fac1 = np.exp(a1 * x_br)
        fac2 = fac1 - 1
        de = (a1 + a2 * fac2)**2  # denominator

        H11 = a2 * (a1**2*fac1*x_br*(2+a2*x_br) - 2*a1*fac2 - a2*fac2**2 - a1**3*fac1*x_br**2) / a1**2 / de
        H12 = (fac1 * (1 - a1 * x_br) - 1) / de
        H13 = (-a2**2*fac2 + a1**2*(1-a2*fac1*x_br) + a1*a2*(fac1*(2+a2*x_br)-2)) / de
        H22 = -a1 * (a1 + 2 * a2 * fac2) / a2**2 / de
        H23 = -a1**2 * fac1 / de
        H33 = -a1**2 * (a1 - a2) * a2 * fac1 / de

        part1 = np.array([
            [zeros+H11, zeros+H12, zeros+H13],
            [zeros+H12, zeros+H22, zeros+H23],
            [zeros+H13, zeros+H23, zeros+H33]
        ])

        H13 = -a2 * fac1 * (a1**2 * x_br + a2 * (fac2 - a1 * x_br)) / de
        H23 = fac2 * (-a1**2 + 2 * a1 * a2 + a2**2 * fac2) / de

        part2 = np.array([
            [zeros+H11, zeros+H12, zeros+H13],
            [zeros+H12, zeros+H22, zeros+H23],
            [zeros+H13, zeros+H23, zeros+H33]
        ])

        part1[..., x >= x_br] = 0.0
        part2[..., x < x_br] = 0.0  # mask out the out-of-scope regimes
        return np.moveaxis((part1 + part2).sum(axis=-1), -1, 0)

    def _fitting_bounds(self, x):
        """
        Return boundaries for fitting based on x (mainly for fitting bootstrap samples)
        :param x: non-negative random variable in this distribution
        """

        bounds = np.copy(self.bounds)
        bounds[2][1] = x[-1]
        return bounds


class TruncatedBrokenPowerLaw(UniVarDistribution):
    """ Distribution class for a truncated broken power law as the PDF.

        This class describes a distribution follows:

               ┏  C_0 (M / M_min)^(-a1)
        p(M) = |
               ┗  C_0 (M / M_min)^(-a2) / (M_br / M_min)^(a2 - a1)
        where a1, a2 stand for alpha_1, alpha_2 for simplicity, and

        C_0 = [ 1/a1 + (1/a2 - 1/a1) * (M_br / M_min)^(-a1)
                - 1/a2 * (M_br / M_min)^(a2 - a1) * (M_tr / M_min)^(-a2) ]^(-1)

        let x = ln(M/M_min), the formula above can be rewritten as:

                ┏  C_0 ( Exp(-a1 x) / a1 + (1/a2 - 1/a1) * Exp(-a1 x_br) - Exp((a2 - a1) x_br - a2 x_tr) / a2 )
        P(>x) = |
                ┗  C_0 Exp((a2 - a1) x_br) * [Exp(-a2 x) - Exp(-a2 x_tr)] / a2

                ┏  C_0 Exp(-a1 x)
        p(x)  = |
                ┗  C_0 Exp(-a2 x + (a2 - a1) x_br)

        C_0 = [ 1/a1 + (1/a2 - 1/a1) * Exp(-a1 x_br) - Exp((a2 - a1) x_br - a2 x_tr) / a2 ]^(-1)
        where the implicit assumptions are
        [-] no constraints on a1
        [-] no constraints on a2
        [-] 0 <= x_br <= x_max <= x_tr (max from data)
    """

    def __init__(self, m, t_guess):

        super().__init__(m, t_guess)

        self.bounds = np.array([[-np.inf, np.inf], [-np.inf, np.inf], [0, self.x[-1]], [self.x[-1], np.inf]])

    def cumulative_func(self, x, t):
        """ Calculate the cumulative distribution value at x (data) given t (parameters)  """

        a1, a2, x_br, x_tr = t  # = [alpha_1, alpha_2, x_br, x_tr] in the theta vector
        part1 = np.exp(-a1 * x) / a1 + (1 / a2 - 1 / a1) * np.exp(-a1 * x_br) - np.exp((a2 - a1) * x_br - a2 * x_tr) / a2
        part2 = np.exp((a2 - a1) * x_br) * (np.exp(-a2 * x) - np.exp(-a2 * x_tr)) / a2
        #mask_x = np.tile(x, [a1.size, 1])
        #mask1 = np.zeros_like(part1); mask1[mask_x < x_br] = 1
        #mask2 = np.zeros_like(part2); mask2[mask_x > x_br] = 1
        #return (part1 * mask1 + part2 * mask2) / (1 / a1 + (1 / a2 - 1 / a1) * np.exp(-a1 * x_br) / a2)
        part1[x >= x_br] = 0.0; part2[x < x_br] = 0.0; part2[x > x_tr] = 0.0
        return (part1 + part2)  / (1 / a1 + (1 / a2 - 1 / a1) * np.exp(-a1 * x_br) - np.exp((a2 - a1) * x_br - a2 * x_tr) / a2)

    def prob_den_func(self, x, t):
        """ Calculate the probability density value at x (data) given t (parameters) """

        a1, a2, x_br, x_tr = t  # = [alpha_1, alpha_2, x_br, x_tr] in the theta vector
        part1 = np.exp(-a1 * x); part2 = np.exp((a2 - a1) * x_br - a2 * x)
        #mask_x = np.tile(x, [a1.size, 1])
        #mask1 = np.zeros_like(part1); mask1[mask_x < x_br] = 1; mask2 = np.zeros_like(part2); mask2[mask_x > x_br] = 1
        #return (part1 * mask1 + part2 * mask2) / (1 / a1 + (1 / a2 - 1 / a1) * np.exp(-a1 * x_br) / a2)
        part1[x >= x_br] = 0.0; part2[x < x_br] = 0.0; part2[x > x_tr] = 0.0
        return (part1 + part2) / (1 / a1 + (1 / a2 - 1 / a1) * np.exp(-a1 * x_br) - np.exp((a2 - a1) * x_br - a2 * x_tr) / a2)

    def jac_func(self, x, t):
        """ Return the Jacobian vector of the truncated broken power law distribution """

        a1, a2, x_br, x_tr = t  # = [alpha_1, alpha_2, x_br, x_tr] in the theta vector
        zeros = np.zeros_like(a1 + x)  # zeros is in fact 2D array
        fac1 = a1 * np.exp(a2 * x_br)
        fac2 = a1 * np.exp(a2 * x_tr)
        fac3 = a2 * np.exp(a2 * x_tr)
        de = a1 + a2 * (np.exp(a1 * x_br) - 1)  # denominator
        part1 = np.array([zeros+(a1*(fac1-fac2)*(x_br-x) + fac3*((1-a1*x)*(1-np.exp(a1*x_br))+a1*x_br)) / (a1*fac1-fac2*de),
                          zeros+(fac2+fac1*(a2*(x_br-x_tr)-1)) / (-a2*fac1+fac3*de),
                          zeros+(a1-a2)*(fac1-fac2) / (fac1-np.exp(a2*x_tr)*de),
                          zeros+(a2*fac1) / (fac1-np.exp(a2*x_tr)*de)])
        part2 = np.array([zeros-fac3*(1+np.exp(a1*x_br)*(a1*x_br-1)) / (-a1*fac1+fac2*de),
                          zeros+((fac2+a2*fac2*(x_br-x)+a2**2*np.exp(a2*x_tr)*(np.exp(a1*x_br)-1)*(x_br-x)) + fac1*(a2*(x-x_tr)-1)) / (-a2*fac1+fac3*de),
                          zeros+(a1-a2)*fac3*(1-np.exp(a1*x_br)) / (-fac1+np.exp(a2*x_tr)*de),
                          zeros+(a2*fac1) / (fac1-np.exp(a2*x_tr)*de)])

        part1[..., x >= x_br] = 0.0
        part2[..., x < x_br] = 0.0  # mask out the out-of-scope regimes
        part2[..., x > x_tr] = 0.0
        return (part1 + part2).sum(axis=-1).T

    def hess_func(self, x, t):
        """ Return the Hessian matrix of the truncated broken power law distribution """

        a1, a2, x_br, x_tr = t  # = [alpha_1, alpha_2, x_br, x_tr] in the theta vector

        zeros = np.zeros_like(a1 + x)  # zeros is in fact 2D array
        fac1 = a1 * np.exp(a2 * x_br)
        fac2 = a1 * np.exp(a2 * x_tr)
        fac3 = a2 * np.exp(a2 * x_tr)
        fac4 = np.exp(a1 * x_br) - 1
        de = (fac3 * (np.exp(a1 * x_br) - 1) + (fac2 - fac1)) ** 2

        H11 = fac3*(-fac3*fac4**2+2*fac4*(fac1-fac2)+a1**2*np.exp(a1*x_br)*(fac1-fac2)*x_br**2+a1*x_br*np.exp(a1*x_br)*(-2*fac1+fac2*(2+a2*x_br))) / (a1**2*de)
        H12 = -(1+np.exp(a1*x_br)*(a1*x_br-1))*(np.exp(a2*x_tr*2)+np.exp(a2*(x_br+x_tr))*(a2*(x_br-x_tr)-1)) / de
        H13 = (fac1-fac2)/a1*(a2*fac3*fac4+a1*(fac1-fac2+fac2*a2*np.exp(a1*x_br)*x_br)-a2*fac2*(-2+np.exp(a1*x_br)*(2+a2*x_br))) / de
        H14 = a2*np.exp(a2*x_br)*fac3*(1+np.exp(a1*x_br)*(a1*x_br-1)) / de
        H22 = ((-(fac1**2+fac2**2-fac1*fac2*(2+a2**2*(x_br-x_tr)**2)))+fac4*(-2*fac3*fac2+fac3*fac1*(2-2*a2*(x_br-x_tr)+a2**2*(x_br-x_tr)**2))) / (a2**2*de)
        H23 = (fac1*fac3*a2*fac4*(x_br-x_tr)-(fac1**2+np.exp(a1*x_br)*fac2**2+np.exp(a1*x_br)*fac1*fac2*(a2*(x_br-x_tr)-1))-fac1*fac2*(a2*(x_tr-x_br)-1)) / de
        H24 = fac1*((a2*fac3*fac4*(x_tr-x_br))+(fac1+fac2*(a2*(x_tr-x_br)-1))) / de
        H33 = a2*(a1-a2)*(-fac1*fac3*fac4-np.exp(a1*x_br)*(fac2**2-fac1*fac2)) / de
        H34 = a2*(a1-a2)*fac1*fac3*fac4 / de
        H44 = a2*fac1*fac3*(a1+a2*fac4) / de

        part1 = np.array([
            [zeros+H11, zeros+H12, zeros+H13, zeros+H14],
            [zeros+H12, zeros+H22, zeros+H23, zeros+H24],
            [zeros+H13, zeros+H23, zeros+H33, zeros+H34],
            [zeros+H14, zeros+H24, zeros+H34, zeros+H44]
        ])

        part2 = np.array([
            [zeros+H11,   zeros+H12,   zeros+H13-1, zeros+H14],
            [zeros+H12,   zeros+H22,   zeros+H23+1, zeros+H24],
            [zeros+H13-1, zeros+H23+1, zeros+H33,   zeros+H34],
            [zeros+H14,   zeros+H24,   zeros+H34,   zeros+H44]
        ])

        part1[..., x >= x_br] = 0.0
        part2[..., x < x_br] = 0.0  # mask out the out-of-scope regimes
        part2[..., x > x_tr] = 0.0
        return np.moveaxis((part1 + part2).sum(axis=-1), -1, 0)

    def _fitting_bounds(self, x):
        """
        Return boundaries for fitting based on x (mainly for fitting bootstrap samples)
        :param x: non-negative random variable in this distribution
        """

        bounds = np.copy(self.bounds)
        bounds[2][1] = x[-1]
        bounds[3][0] = x[-1]
        return bounds


class ThreeSegPowerLaw(UniVarDistribution):
    """ Distribution class for a three-segment power law as the PDF.

        This class describes a distribution follows:

               ┏  C_1 (M / M_min)^(-a1)
        p(M) = |  C_1 (M / M_min)^(-a2) / (M_br1 / M_min)^(a1 - a2)
               ┗  C_1 (M / M_min)^(-a2) / (M_br1 / M_min)^(a1 - a2) / (M_br2 / M_min)^(a2 - a3)
        where a1, a2, a3 stand for alpha_1, alpha_2, alpha_3 for simplicity, and

        C_1 = [ 1/a1 + (1/a2 - 1/a1) * (M_br1 / M_min)^(-a1)
                + (1/a3 - 1/a2) * (M_br1 / M_min)^(a2 - a1) * (M_br2 / M_min)^(-a2) ]^(-1)

        let x = ln(M/M_min), the formula above can be rewritten as:

                ┏  C_1 ( Exp(-a1 x)/a1 + (1/a2 - 1/a1) Exp(-a1 x_br1) + (1/a3 - 1/a2) Exp((a2 - a1) x_br1 - a2 x_br2)) )
        P(>x) = |  C_1 ( Exp((a2 - a1) x_br1 - a2 x) / a2 + (1/a3 - 1/a2) Exp((a2 - a1) x_br1 - a2 x_br2) )
                ┗  C_1 Exp((a2 - a1) x_br1 + (a3 - a2) x_br2 - a3 x) / a3

                ┏  C_1 Exp(-a1 x)
        p(x)  = |  C_1 Exp((a2 - a1) x_br1 - a2 x)
                ┗  C_1 Exp((a2 - a1) x_br1 + (a3 - a2) x_br2 - a3 x)

        C_0 = [ 1/a1 + (1/a2 - 1/a1) * Exp(-a1 x_br1) + (1/a3 - 1/a2) * Exp((a2-a1) x_br1 -a2 x_br2) ]^(-1)
        where the implicit assumptions are
        [-] no constraints on a1 or a2
        [-] a3 > 0
        [-] 0 <= x_br1 <= x_br2 <= x_max (max from data)
    """

    def __init__(self, m, t_guess):

        super().__init__(m, t_guess)

        self.bounds = np.array([[-np.inf, np.inf], [-np.inf, np.inf], [0, np.inf], [0, self.x[-1]], [0, self.x[-1]]])

    def cumulative_func(self, x, t):
        """ Calculate the cumulative distribution value at x (data) given t (parameters)  """

        a1, a2, a3, x_br1, x_br2 = t  # = [alpha_1, alpha_2, alpha_3, x_br1, x_br2] in the theta vector
        part1 = np.exp(-a1 * x) / a1 + (1/a2 - 1/a1) * np.exp(-a1 * x_br1) \
                + (1 / a3 - 1 / a2) * np.exp((a2 - a1) * x_br1 - a2 * x_br2)
        part2 = np.exp((a2 - a1) * x_br1 - a2 * x) / a2 + (1/a3 - 1/a2) * np.exp((a2 - a1) * x_br1 - a2 * x_br2)
        part3 = np.exp((a2 - a1) * x_br1 + (a3 - a2) * x_br2 - a3 * x) / a3

        part1[x >= x_br1] = 0.0; part2[(x < x_br1) | (x >= x_br2)] = 0.0; part3[x < x_br2] = 0.0

        return (part1 + part2 + part3) / (1/a1 + (1/a2 - 1/a1) * np.exp(-a1 * x_br1)
                                          + (1/a3 - 1/a2) * np.exp((a2 - a1) * x_br1 - a2 * x_br2))

    def prob_den_func(self, x, t):
        """ Calculate the probability density value at x (data) given t (parameters) """

        a1, a2, a3, x_br1, x_br2 = t  # = [alpha_1, alpha_2, alpha_3, x_br1, x_br2] in the theta vector
        part1 = np.exp(-a1 * x)
        part2 = np.exp((a2 - a1) * x_br1 - a2 * x)
        part3 = np.exp((a2 - a1) * x_br1 + (a3 - a2) * x_br2 - a3 * x)
        part1[x >= x_br1] = 0.0; part2[(x < x_br1) | (x >= x_br2)] = 0.0; part3[x < x_br2] = 0.0
        return (part1 + part2 + part3) / (1 / a1 + (1 / a2 - 1 / a1) * np.exp(-a1 * x_br1)
                                          + (1 / a3 - 1 / a2) * np.exp((a2 - a1) * x_br1 - a2 * x_br2))

    def ln_prob(self, t, x):
        """
        Calculate the log-likelihood of data (x) given the distribution parameters (t)
        :param x: non-negative random variable in this distribution
        :param t: distribution parameters (vector theta); can accept an array of vector theta
        """

        t = np.asarray(t)
        L = np.atleast_1d(np.sum(np.log(self.pdf(x, t=t)), axis=t.ndim - 1))
        L[np.isnan(L)] = -np.inf
        # the customized bounds serve for the purpose of ln_prior
        bounds = self._fitting_bounds(x)
        if bounds is not None:
            x_br2_lower_bound = 0
            for i, item in enumerate([np.atleast_1d(_t) for _t in t.T]):
                if i == 3:
                    x_br2_lower_bound = item
                if i == 4:  # ensure x_br2 >= x_br1
                    L[(item < x_br2_lower_bound) | (item > bounds[i][1])] = -np.inf
                else:
                    L[(item < bounds[i][0]) | (item > bounds[i][1])] = -np.inf
        return np.squeeze(L)

    def jac_func(self, x, t):
        """ Return the Jacobian vector of the three-segment power law distribution """

        a1, a2, a3, x_br1, x_br2 = t  # = [alpha_1, alpha_2, alpha_3, x_br1, x_br2] in the theta vector
        fac1 = np.exp(a1 * x_br1)
        fac2 = fac1 - 1
        fac3 = np.exp(a2 * x_br1)
        fac4 = np.exp(a2 * x_br2)
        fac5 = a1 * (a2 - a3) * fac3
        fac6 = a3 * fac4 * (a1 + a2 * fac2)
        de = fac5 + fac6
        zeros = np.zeros_like(a1 + x)
        J1 = zeros + (-a2*a3*fac4*(1+fac1*(a1*x_br1-1))) / (a1 * de)
        J3 = zeros + a1*a2*fac3 / (a3 * de)
        J4 = zeros + (-(a1-a2)*a2*a3*fac4*fac2 / de)
        J5 = zeros + a1*a2*(a2-a3)*fac3 / de

        part1 = np.array([
            (a1*fac5*(x_br1-x) - a3*fac4*(a1**2*(x-x_br1)+a2*(1-a1*x+fac1*(a1*x-1)+a1*x_br1))) / (a1 * de),
            zeros + (a1*a3*fac4 - a1*fac3*(a3+a2**2*(x_br1-x_br2)+a2*a3*(x_br2-x_br1))) / (a2 * de),
            J3,
            zeros + a1*(a1-a2)*((a2-a3)*fac3+a3*fac4) / de,
            J5
        ])
        part2 = np.array([
            J1,
            (a3*fac4*(a1+a1*a2*(x_br1-x)+a2**2*fac2*(x_br1-x))
             - a1*fac3*(a3+a2**2*(x-x_br2)+a2*a3*(x_br2-x))) / (a2 * de),
            J3,
            J4,
            J5
        ])
        part3 = np.array([
            J1,
            zeros + (-a1*a3*fac3+a3*fac4*(a1+a1*a2*(x_br1-x_br2)+a2**2*fac2*(x_br1-x_br2))) / (a2 * de),
            J3 - x + x_br2,
            J4,
            zeros + (a2-a3)*a3*(a1*fac3-fac4*(a1+a2*fac2)) / de
        ])
        part1[..., x >= x_br1] = 0.0
        part2[..., (x < x_br1) | (x >= x_br2)] = 0.0
        part3[..., x < x_br2] = 0.0
        return (part1 + part2 + part3).sum(axis=-1).T

    def hess_func(self, x, t):
        """ Return the Hessian matrix of the three-segment power law distribution """

        a1, a2, a3, x_br1, x_br2 = t  # = [alpha_1, alpha_2, alpha_3, x_br1, x_br2] in the theta vector

        raise NotImplementedError("The calculation of Hessian matrix for the three-segment power law \
         distribution has not been implemented ")

    def _fitting_bounds(self, x):
        """
        Return boundaries for fitting based on x (mainly for fitting bootstrap samples)
        :param x: non-negative random variable in this distribution
        """

        bounds = np.copy(self.bounds)
        bounds[3][1] = x[-1]
        bounds[4][1] = x[-1]
        return bounds

    def test_mini_method(self, t_guess, x, tol=range(8, 9), max_f_i=int(2e5), more_methods=False, methods=None,
                         ln_prob_format=r'{:.2f}', t_format=r'{:+.2e}', no_warnings=True, disp=False):
        """
        Test various minimization method with scipy.optimize.minimize
        :param t_guess: initial guess for the distribution parameters (vector theta)
        :param x: non-negative random variable in this distribution
        :param tol: array, list or tuple, all kinds of tolerance for successful termination
        :param max_f_i: scalar, maximum number of function evaluation / iteration
        :param more_methods: bool, whether or not to include more methods in testing
        :param methods: a list (or tuple) of strings, customized methods by users
        :param ln_prob_format: formatter for -ln-L
        :param t_format: formatter for parameters
        :param no_warnings: disable warnings
        :param disp: whether to add disp into minimization options
        """

        if methods is None:
            all_methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC']
            if more_methods:
                all_methods.append(['dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact'])
        else:
            all_methods = methods
        bounds = self._fitting_bounds(x)

        with warnings.catch_warnings():
            if no_warnings: warnings.simplefilter('ignore')
            if tol is None:
                if max_f_i is None:
                    print("Use default minimization options")
                else:
                    print("Use default tolerance with maxfev/maxfun/maxiter=", max_f_i)
                for meth in all_methods:
                    if max_f_i is None:
                        res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess, method=meth, bounds=bounds,
                                             jac=lambda t: -self.grad_likelihood(x, t=t), options={'disp': disp})
                    else:
                        res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess, method=meth, bounds=bounds,
                                             jac=lambda t: -self.grad_likelihood(x, t=t),
                                             options={'maxfev': max_f_i, 'maxfun': max_f_i,
                                                      'maxiter': max_f_i, 'disp': disp})
                    print((r'method: {:<11} success? {:1} -lnL= '+ln_prob_format+r', grad-L= ['
                          +r'{:+.2e}, '*(self.t.size-1)+r'{:+.2e}] t = ['+(t_format+r', ')*(self.t.size-1)
                          +t_format+']').format(meth, res.success, res.fun, *tuple(self.grad_likelihood(x, t=res.x)),
                                                *tuple(res.x)), flush=True)
            else:
                for ep in tol:
                    print("Now tolerance = 1e-"+str(ep), flush=True)
                    for meth in all_methods:
                        res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess, method=meth, bounds=bounds,
                                             jac=lambda t: -self.grad_likelihood(x, t=t),
                                             options={'maxfev': max_f_i, 'maxfun': max_f_i, 'maxiter': max_f_i,
                                                      'xtol': 10**(-ep), 'gtol': 10**(-ep), 'ftol': 10**(-ep),
                                                      'xatol': 10**(-ep), 'fatol': 10**(-ep), 'disp': disp})
                        print((r'method: {:<11} success? {:1} -lnL= '+ln_prob_format+r', grad-L= ['
                              +r'{:+.2e}, '*(self.t.size-1)+r'{:+.2e}] t = ['+(t_format+r', ')*(self.t.size-1)
                              +t_format+']').format(meth, res.success, res.fun,
                                                    *tuple(self.grad_likelihood(x, t=res.x)), *tuple(res.x)),
                              flush=True)

    def _mesh_grid_mini(self, method, x, t_guess, tol=12, max_f_i=int(2e5)):
        """
        sub-method for mesh grid minimization in parameter space
        :param method: which method for scipy.optimize.minimize
        :param x: non-negative random variable in this distribution
        :param t_guess: initial guess for the distribution parameters (vector theta)
        :param tol: array, list or tuple, all kinds of tolerance for successful termination
        :param max_f_i: scalar, maximum number of function evaluation / iteration
        """

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess, method=method, bounds=self._fitting_bounds(x),
                                 jac=lambda t: -self.grad_likelihood(x, t=t),
                                 options={'maxfev': max_f_i, 'maxfun': max_f_i, 'maxiter': max_f_i,
                                          'xtol': 10**(-tol), 'gtol': 10**(-tol), 'ftol': 10**(-tol),
                                          'xatol': 10**(-tol), 'fatol': 10**(-tol)})
            return np.hstack([res.fun, res.x])

    def _minus_ln_prob_min(self, t_guess, x, method, silent=True):
        """
        Minimize the minus log likelihood by scipy.optimize.minimize function
        :param t_guess: initial guess for the distribution parameters (vector theta)
        :param x: non-negative random variable in this distribution
        :param method: minimization method for scipy
        """

        bounds = self._fitting_bounds(x)
        if bounds is not None:
            t_guess = np.maximum(t_guess, bounds[:, 0])
            t_guess = np.minimum(t_guess, bounds[:, 1])

        if method in ['Nelder-Mead', 'Powell']:
            if self.mini_options is None: self.mini_options = self._mini_default_options(method)
            if not silent: self.mini_options['disp'] = True
            res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess, method=method,
                                 options=self.mini_options)
        elif method in ['BFGS', 'CG']:
            if self.mini_options is None: self.mini_options = self._mini_default_options(method)
            if not silent: self.mini_options['disp'] = True
            res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess, method=method,
                                 jac=lambda t: -self.grad_likelihood(x, t=t), options=self.mini_options)
        elif method in ['L-BFGS-B', 'TNC']:
            if self.mini_options is None: self.mini_options = self._mini_default_options(method)
            if not silent: self.mini_options['disp'] = True
            res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess, method=method, bounds=bounds,
                                 jac=lambda t: -self.grad_likelihood(x, t=t), options=self.mini_options)
        elif method in ['Newton-CG']:
            if self.mini_options is None: self.mini_options = self._mini_default_options(method)
            if not silent: self.mini_options['disp'] = True
            res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess, method=method, bounds=bounds,
                                 jac=lambda t: -self.grad_likelihood(x, t=t), options=self.mini_options)
        else:
            if self.mini_options is None and (not silent): self.mini_options = {'disp': True}
            res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess, method=method, bounds=bounds,
                                 jac=lambda t: -self.grad_likelihood(x, t=t), options=self.mini_options)
        return res
