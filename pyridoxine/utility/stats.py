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
#from IPython.display import display, Math, Latex


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
        where the implicit assumptions is alpha > 0, beta > 0, 0 < x_exp < x_max (max from data).

        To construct a new distribution model, you may want to overwrite:
        [-] self.cumulative_func
        [-] self.prob_den_func
        [-] self.jac_func (optional, but needed by self.minus_ln_prob_min if using BFGS)
        [-] self.hess_func (optional)
        Also, don't forget to apply appropriate self.bounds.
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

        self.bounds = None  # sequence of (min, max) pairs for each parameter for evaluating ln_prob

        # For MCMC likelihood exploration
        self.mcmc_dim = self.t.size
        self.mcmc_num_walkers = 32
        self.mcmc_num_steps = 10000
        self.mcmc_burn_in = 1000
        self.sln_mcmc = self.t

        # For minus likelihood minimization with scipy
        self.mini_method = 'Nelder-Mead'
        self.mini_options = None

        # For bootstrapping
        self.bs_samples = None
        self.bs_t = None
        self.bs_t_std = None
        self.bs_likelihood = 0

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
        if self.bounds is not None:
            for i, item in enumerate([np.atleast_1d(_t) for _t in t.T]):
                L[(item<self.bounds[i][0])|(item>self.bounds[i][1])] = -np.inf
        return np.squeeze(L)

    def mcmc_fitting(self, x, t_guess=None, **kwargs):
        """
        Do MCMC fitting to explore the parameter space
        :param x: non-negative random variable in this distribution
        :param t_guess: initial guess for the distribution parameters (vector theta)
        :param kwargs: other keywords for do_mcmc
        """
        if self.bounds is None:
            Warning("No boundary has been set for parameters.")
        return do_mcmc(self.mcmc_dim, self.mcmc_num_walkers, self.mcmc_num_steps, self.mcmc_burn_in,
                       (self.t if t_guess is None else np.asarray(t_guess)),
                       self.ln_prob, x, vectorize=True, **kwargs)

    def minus_ln_prob_min(self, t_guess, x, silent=True):
        """
        Minimize the minus log likelihood by scipy.optimize.minimize function
        :param t_guess: initial guess for the distribution parameters (vector theta)
        :param x: non-negative random variable in this distribution
        Different method yield different options
        """

        if self.mini_method in ['Nelder-Mead']:
            if self.mini_options is None:
                self.mini_options = {'disp': True, 'xatol': 1e-15, 'fatol': 1e-15, 'maxfev': 1e5}
            res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess,
                                 method=self.mini_method, options=self.mini_options)
        if self.mini_method in ['Powell']:
            if self.mini_options is None:
                self.mini_options = {'disp': True, 'xtol': 1e-15, 'ftol': 1e-15, 'maxfev': 1e5}
            res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess,
                                 method=self.mini_method, options=self.mini_options)
        elif self.mini_method in ['L-BFGS-B']:
            if self.bounds is None:
                raise RuntimeError("L-BFGS-B minimization needs boundaries for parameters")
            if self.mini_options is None:
                self.mini_options = {'disp': True, 'ftol': 1e-15, 'gtol': 1e-15, 'maxfun': 10000}
            res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess, method=self.mini_method,
                                 jac=lambda t: -self.grad_likelihood(x, t=t), bounds=self.bounds,
                                 options=self.mini_options)
        elif self.mini_method in ['TNC']:
            if self.bounds is None:
                raise RuntimeError("TNC minimization needs boundaries for parameters")
            if self.mini_options is None:
                self.mini_options = {'disp': True, 'ftol': 1e-15, 'gtol': 1e-15, 'xtol': 1e-15, 'maxiter': 10000}
            res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess, method=self.mini_method,
                                 jac=lambda t: -self.grad_likelihood(x, t=t), bounds=self.bounds,
                                 options=self.mini_options)
        elif self.mini_method in ['BFGS']:
            if self.mini_options is None:
                self.mini_options = {'disp': True, 'gtol': 1e-15}
            res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess, method = self.mini_method,
                                 jac = lambda t: -self.grad_likelihood(x, t=t), bounds = self.bounds,
                                 options = self.mini_options)
        else:
            if self.mini_options is None:
                self.mini_options = {'disp': True}
            res = spopt.minimize(lambda t: -self.ln_prob(t, x), t_guess,
                                 method=self.mini_method, options=self.mini_options)
        if not res.success:
            Warning(r'Optimization terminated unsuccessfully')
        if not silent:
            print("minimization results:\n", res)
        return res.x  # this is automatically numpy.ndarray

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
        self.mini_options.pop("disp", None)
        p = Pool(threads)
        tmp_func = partial(self.minus_ln_prob_min, t_guess)
        bs_t = np.array(p.map(tmp_func, bs_samples))
        p.close()
        bs_likelihood = self.ln_prob(bs_t, self.x)
        bs_t = bs_t[(~np.isinf(bs_likelihood))&(~np.isnan(bs_likelihood))]
        bs_likelihood = bs_likelihood[(~np.isinf(bs_likelihood))&(~np.isnan(bs_likelihood))]
        if bs_likelihood.size < n_bs:
            Warning('Dropping {:d} samples with unaccepted theta or -inf likelihood'.format(n_bs - bs_likelihood.size))
        self.bs_t, self.bs_t_std, self.bs_likelihood = bs_t, bs_t.std(axis=0), -bs_likelihood.mean()
        self.mini_options["disp"] = True
        return self.bs_t_std, self.bs_likelihood

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
            Warning('Dropping {:d} samples with unaccepted theta'.format(n_bs - bs_t.shape[0]))
        bs_likelihood = self.ln_prob(bs_t, self.x)
        bs_t = bs_t[(~np.isinf(bs_likelihood)) & (~np.isnan(bs_likelihood))]
        bs_likelihood = bs_likelihood[(~np.isinf(bs_likelihood)) & (~np.isnan(bs_likelihood))]
        if bs_likelihood.size < n_bs:
            Warning('Dropping {:d} samples with unaccepted theta or -inf likelihood'.format(n_bs - bs_likelihood.size))
        self.bs_t, self.bs_t_std, self.bs_likelihood = bs_t, bs_t.std(axis=0), -bs_likelihood.mean()
        return self.bs_t_std, self.bs_likelihood

    def fitting(self, use_solver=False, jac=False, inv_hess=False, silent=False, f_format="{:.8e}", **kwargs):
        """
        Perform the MCMC exploration and -ln_prob minimization
        :param use_solver: solve the maximum likelihood equation set instead of MCMC&ln_prob
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
            self.sln_mcmc = self.mcmc_fitting(self.x, silent=False, **kwargs)
            print("Chosen minimization method: ", self.mini_method)
            self.t = self.minus_ln_prob_min(self.sln_mcmc, self.x, silent=False)

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
        where the implicit assumptions is alpha > 0, 0 < x_exp < x_max (max from data).
    """

    def __init__(self, m, t_guess):

        super().__init__(m, t_guess)

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


class TruncatedPowerLaw(UniVarDistribution):
    """ Distribution class for a truncated power law.

        This class describes a distribution follows:

        p(M) = (a / M) * (M / M_min)^(-a) / (1 - (M_max / M_min)^(-a)), 

        where a stands for alpha for simplicity.

        let x = ln(M/M_min), the formula above can be rewritten as:
        P(>x) = (1 - Exp(a * (x_max - x))) / (1 - Exp(a x_max))
        p(x)  = a Exp(-a x) / (1 - Exp(-a x_max))
        where the implicit assumptions is alpha > 0, x_max should >= ln(M_max/M_min) in data.
    """

    def __init__(self, m, t_guess):

        super().__init__(m, t_guess)

        self.cumulative_func = lambda x, t: (1 - np.exp(t[0] * (t[1] - x))) / (1 - np.exp(t[0] * t[1]))
        self.prob_den_func = lambda x, t: (t[0] * np.exp(-t[0] * x)) / (1 - np.exp(-t[0] * t[1]))

    def jac_func(self, x, t):
        """ Return the Jacobian vector of the truncated power law distribution """

        a, x_max = t  # = [alpha, x_max] in the vector theta
        de = 1 - np.exp(a * x_max)
        return np.array([1/a - x + x_max / de, np.zeros_like(a + x) + a / de]).sum(axis=-1).T

    def hess_func(self, x, t):
        """ Return the Hessian matrix of the truncated power law distribution """

        a, x_max = t  # = [alpha, x_max] in the vector theta
        de = (1 - np.exp(a * x_max))**2
        zeros = np.zeros_like(a + x)
        return np.moveaxis(np.array([
            [zeros + (- 1/a**2 + x_max**2 / 4 * 1/np.sinh(a * x_max / 2)**2),
             zeros + (1 + np.exp(a * x_max) * (a * x_max - 1)) / de],
            [zeros + (1 + np.exp(a * x_max) * (a * x_max - 1)) / de,
             zeros + a**2 / 4 * 1/np.sinh(a * x_max / 2)**2]
        ]).sum(axis=-1), -1, 0)


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

        where the implicit assumptions is 0 < x_br < x_max (max from data)
    """

    def __init__(self, m, t_guess):

        super().__init__(m, t_guess)

    def cumulative_func(self, x, t):
        """ Calculate the cumulative distribution value at x (data) given t (parameters)  """

        a1, a2, x_br = t  # = [alpha_1, alpha_2, x_br] in the theta vector
        part1 = np.exp(-a1*x); part2 = np.exp((a2-a1)*x_br-a2*x)
        # old method:
        #mask_x = np.tile(x, [a1.size, 1])
        #mask1 = np.zeros_like(part1); mask1[mask_x<x_br] = 1; mask2 = np.zeros_like(part2); mask2[mask_x>x_br] = 1
        #return part1 * mask1 + part2 * mask2
        # in fact, (x > x_br) has the same shape as part1
        part1[x > x_br] = 0.0; part2[x < x_br] = 0.0
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
        part1[x > x_br] = 0.0; part2[x < x_br] = 0.0
        return part1 + part2

    def jac_func(self, x, t):
        """ Return the Jacobian vector of the broken cumulative power law distribution """

        a1, a2, x_br = t  # = [alpha_1, alpha_2, x_br] in the theta vector
        zeros = np.zeros_like(a1 + x)  # zeros is in fact 2D array
        part1 = np.array([1/a1-x, zeros, zeros]); part2 = np.array([zeros-x_br, 1/a2+x_br-x, zeros-a1+a2])
        # N.B., (x > x_br) is two dimensional, because x_br is a vertical vector
        # thus, part1[:, x > x_br] will pin down the correct items.
        # Alternatively, use part1[..., x > x_br] to select unwanted items.
        part1[:, x > x_br] = 0.0; part2[:, x < x_br] = 0.0  # mask out the out-of-scope regimes
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
        part1[:, :, x > x_br] = 0.0; part2[:, :, x < x_br] = 0.0  # mask out the out-of-scope regimes
        return np.moveaxis((part1 + part2).sum(axis=-1), -1, 0)

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
        where the implicit assumptions is 0 < x_br < x_max (max from data)
    """

    def __init__(self, m, t_guess):

        super().__init__(m, t_guess)

    def cumulative_func(self, x, t):
        """ Calculate the cumulative distribution value at x (data) given t (parameters)  """

        a1, a2, x_br = t  # = [alpha_1, alpha_2, x_br] in the theta vector
        part1 = np.exp(-a1 * x) / a1 + (1 / a2 - 1 / a1) * np.exp(-a1 * x_br)
        part2 = np.exp((a2 - a1) * x_br - a2 * x) / a2
        #mask_x = np.tile(x, [a1.size, 1])
        #mask1 = np.zeros_like(part1); mask1[mask_x < x_br] = 1
        #mask2 = np.zeros_like(part2); mask2[mask_x > x_br] = 1
        #return (part1 * mask1 + part2 * mask2) / (1 / a1 + (1 / a2 - 1 / a1) * np.exp(-a1 * x_br) / a2)
        part1[x > x_br] = 0.0; part2[x < x_br] = 0.0
        return (part1 + part2)  / (1 / a1 + (1 / a2 - 1 / a1) * np.exp(-a1 * x_br))

    def prob_den_func(self, x, t):
        """ Calculate the probability density value at x (data) given t (parameters) """

        a1, a2, x_br = t  # = [alpha_1, alpha_2, x_br] in the theta vector
        part1 = np.exp(-a1 * x); part2 = np.exp((a2 - a1) * x_br - a2 * x)
        #mask_x = np.tile(x, [a1.size, 1])
        #mask1 = np.zeros_like(part1); mask1[mask_x < x_br] = 1; mask2 = np.zeros_like(part2); mask2[mask_x > x_br] = 1
        #return (part1 * mask1 + part2 * mask2) / (1 / a1 + (1 / a2 - 1 / a1) * np.exp(-a1 * x_br) / a2)
        part1[x > x_br] = 0.0; part2[x < x_br] = 0.0
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

        part1[..., x > x_br] = 0.0
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

        part1[..., x > x_br] = 0.0
        part2[..., x < x_br] = 0.0  # mask out the out-of-scope regimes
        return np.moveaxis((part1 + part2).sum(axis=-1), -1, 0)


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
        where the implicit assumptions is 0 < x_br1 < x_br2 < x_max (max from data)
    """

    def __init__(self, m, t_guess):

        super().__init__(m, t_guess)

    def cumulative_func(self, x, t):
        """ Calculate the cumulative distribution value at x (data) given t (parameters)  """

        a1, a2, a3, x_br1, x_br2 = t  # = [alpha_1, alpha_2, alpha_3, x_br1, x_br2] in the theta vector
        part1 = np.exp(-a1 * x) / a1 + (1/a2 - 1/a1) * np.exp(-a1 * x_br1) \
                + (1 / a3 - 1 / a2) * np.exp((a2 - a1) * x_br1 - a2 * x_br2)
        part2 = np.exp((a2 - a1) * x_br1 - a2 * x) / a2 + (1/a3 - 1/a2) * np.exp((a2 - a1) * x_br1 - a2 * x_br2)
        part3 = np.exp((a2 - a1) * x_br1 + (a3 - a2) * x_br2 - a3 * x) / a3

        part1[x > x_br1] = 0.0; part2[(x < x_br1) | (x > x_br2)] = 0.0; part3[x < x_br2] = 0.0

        return (part1 + part2 + part3) / (1/a1 + (1/a2 - 1/a1) * np.exp(-a1 * x_br1)
                                          + (1/a3 - 1/a2) * np.exp((a2 - a1) * x_br1 - a2 * x_br2))

    def prob_den_func(self, x, t):
        """ Calculate the probability density value at x (data) given t (parameters) """

        a1, a2, a3, x_br1, x_br2 = t  # = [alpha_1, alpha_2, alpha_3, x_br1, x_br2] in the theta vector
        part1 = np.exp(-a1 * x)
        part2 = np.exp((a2 - a1) * x_br1 - a2 * x)
        part3 = np.exp((a2 - a1) * x_br1 + (a3 - a2) * x_br2 - a3 * x)
        part1[x > x_br1] = 0.0; part2[(x < x_br1) | (x > x_br2)] = 0.0; part3[x < x_br2] = 0.0
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
        if self.bounds is not None:
            x_br2_lower_bound = 0
            for i, item in enumerate([np.atleast_1d(_t) for _t in t.T]):
                if i == 3:
                    x_br2_lower_bound = item
                if i == 4:  # ensure x_br2 > x_br1
                    L[(item < x_br2_lower_bound) | (item > self.bounds[i][1])] = -np.inf
                else:
                    L[(item < self.bounds[i][0]) | (item > self.bounds[i][1])] = -np.inf
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
        part1[..., x > x_br1] = 0.0
        part2[..., (x < x_br1) | (x > x_br2)] = 0.0
        part3[..., x < x_br2] = 0.0
        return (part1 + part2 + part3).sum(axis=-1).T

    def hess_func(self, x, t):
        """ Return the Hessian matrix of the three-segment power law distribution """

        a1, a2, a3, x_br1, x_br2 = t  # = [alpha_1, alpha_2, alpha_3, x_br1, x_br2] in the theta vector

        raise NotImplementedError("The calculation of Hessian matrix for the three-segment power law \
         distribution has not been implemented ")
