
.. raw:: html

   <div id="qe-notebook-header" align="right" style="text-align:right;">

::

       <a href="https://quantecon.org/" title="quantecon.org">
               <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
       </a>

.. raw:: html

   </div>

Addition to Job Search with Learning lectures
=============================================

In addition to what’s in Anaconda, this lecture will need the following
libraries:

.. code-block:: python3

    !pip install interpolation

.. code-block:: ipython

    from numba import njit, prange, vectorize
    from interpolation import mlinterp, interp
    from math import gamma
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline
    from matplotlib import cm


In the following, we first define two functions for computing the
empirical distributions of unemployment duration and π at the time of
employment.

.. code-block:: python3

    @njit
    def empirical_dist(F_a, F_b, G_a, G_b, w_bar, π_grid,
                       N=10000, T=600):
        """
        Simulates population for computing empirical cumulative
        distribution of unempoyment duration and π at time when
        the worker accepts the wage offer. For each job searching
        problem, we simulate for two cases that either f or g is
        the true offer distribution.
    
        Parameters
        ----------
    
        F_a, F_b, G_a, G_b : parameters of beta distributions F and G.
        w_bar : the reservation wage
        π_grid : grid points of π, for interpolation
        N : number of workers for simulation, optional
        T : maximum of time periods for simulation, optional
    
        Returns
        -------
        accpet_t : 2 by N ndarray. the empirical distribution of 
                   unemployment duration when f or g generates offers.
        accept_π : 2 by N ndarray. the empirical distribution of 
                   π at the time of employment when f or g generates offers.
        """
    
        accept_t = np.empty((2, N))
        accept_π = np.empty((2, N))
    
        # f or g generates offers
        for i, (a, b) in enumerate([(F_a, F_b), (G_a, G_b)]):
            # update each agent
            for n in range(N):
    
                # initial priori
                π = 0.5
    
                for t in range(T+1):
    
                    # Draw random wage
                    w = np.random.beta(a, b)
                    lw = p(w, F_a, F_b) / p(w, G_a, G_b)
                    π = π * lw / (π * lw + 1 - π)
    
                    # move to next agent if accepts
                    if w >= interp(π_grid, w_bar, π):
                        break
    
                # record the unemployment duration
                # and π at the time of acceptance
                accept_t[i, n] = t
                accept_π[i, n] = π
    
        return accept_t, accept_π
    
    def cumfreq_x(res):
        """
        A helper function for calculating the x grids of
        the cumulative frequency histogram.
        """
    
        cumcount = res.cumcount
        lowerlimit, binsize = res.lowerlimit, res.binsize
    
        x = lowerlimit + np.linspace(0, binsize*cumcount.size, cumcount.size)
    
        return x

Now we define a wrapper function for analyzing job search models with
learning under different parameterizations.

It takes parameters of beta distributions and the unemployment
compensation as inputs, and then displays various things we want to know
to interpret the solution of our search model

In addition, it computes empirical cumulative distributions.

.. code-block:: python3

    def job_search_example(F_a=1, F_b=1, G_a=3, G_b=1.2, c=0.3):
        """
        Given the parameters that specify F and G distributions,
        calculate and display the rejection and acceptance area,
        the evolution of belief π, and the probabiilty of accepting
        an offer at different π level, and simulate and calculate
        the empirical cumulative distribution of the duration of
        unemployment and π at the time the worker accepts the offer.
        """
    
        # construct a search problem
        sp = SearchProblem(F_a=F_a, F_b=F_b, G_a=G_a, G_b=G_b, c=c)
        f, g = sp.f, sp.g
        π_grid = sp.π_grid
    
        # Solve for reservation wage
        w_bar = solve_wbar(sp, verbose=False)
    
        # l(w) = f(w) / g(w)
        l = lambda w: f(w) / g(w)
        # objective function for solving l(w) = 1
        obj = lambda w: l(w) - 1.
    
        # the mode of beta distribution
        # use this to divide w into two intervals for root finding
        G_mode = (G_a - 1) / (G_a + G_b - 2)
        roots = np.empty(2)
        roots[0] = op.root_scalar(obj, bracket=[1e-10, G_mode]).root
        roots[1] = op.root_scalar(obj, bracket=[G_mode, 1-1e-10]).root
    
        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    
        # part 1: display the details of the model settings and some results
        w_grid = np.linspace(1e-12, 1-1e-12, 100)
    
        axs[0, 0].plot(l(w_grid), w_grid, label='$l$', lw=2)
        axs[0, 0].vlines(1., 0., 1., linestyle="--")
        axs[0, 0].hlines(roots, 0., 2., linestyle="--")
        axs[0, 0].set_xlim([0., 2.])
        axs[0, 0].legend(loc=4)
        axs[0, 0].set(xlabel='$l(w)=f(w)/g(w)$', ylabel='$w$')
    
        axs[0, 1].plot(sp.π_grid, w_bar, color='k')
        axs[0, 1].fill_between(sp.π_grid, 0, w_bar, color='blue', alpha=0.15)
        axs[0, 1].fill_between(sp.π_grid, w_bar, sp.w_max, color='green', alpha=0.15)
        axs[0, 1].text(0.5, 0.6, 'reject')
        axs[0, 1].text(0.7, 0.9, 'accept')
    
        W = np.arange(0.01, 0.99, 0.08)
        Π = np.arange(0.01, 0.99, 0.08)
    
        ΔW = np.zeros((len(W), len(Π)))
        ΔΠ = np.empty((len(W), len(Π)))
        for i, w in enumerate(W):
            for j, π in enumerate(Π):
                lw = l(w)
                ΔΠ[i, j] = π * (lw / (π * lw + 1 - π) - 1)
    
        q = axs[0, 1].quiver(Π, W, ΔΠ, ΔW, scale=2, color='r', alpha=0.8)
    
        axs[0, 1].hlines(roots, 0., 1., linestyle="--")
        axs[0, 1].set(xlabel='$\pi$', ylabel='$w$')
        axs[0, 1].grid()
    
        axs[1, 0].plot(f(x_grid), x_grid, label='$f$', lw=2)
        axs[1, 0].plot(g(x_grid), x_grid, label='$g$', lw=2)
        axs[1, 0].vlines(1., 0., 1., linestyle="--")
        axs[1, 0].hlines(roots, 0., 2., linestyle="--")
        axs[1, 0].legend(loc=4)
        axs[1, 0].set(xlabel='$f(w), g(w)$', ylabel='$w$')
    
        axs[1, 1].plot(sp.π_grid, 1 - beta.cdf(w_bar, F_a, F_b), label='$f$')
        axs[1, 1].plot(sp.π_grid, 1 - beta.cdf(w_bar, G_a, G_b), label='$g$')
        axs[1, 1].set_ylim([0., 1.])
        axs[1, 1].grid()
        axs[1, 1].legend(loc=4)
        axs[1, 1].set(xlabel='$\pi$', ylabel='$\mathbb{P}\{w > \overline{w} (\pi)\}$')
    
        plt.show()
    
        # part 2: simulate empirical cumulative distribution
        accept_t, accept_π = empirical_dist(F_a, F_b, G_a, G_b, w_bar, π_grid)
        N = accept_t.shape[1]
    
        cfq_t_F = cumfreq(accept_t[0, :], numbins=100)
        cfq_π_F = cumfreq(accept_π[0, :], numbins=100)
    
        cfq_t_G = cumfreq(accept_t[1, :], numbins=100)
        cfq_π_G = cumfreq(accept_π[1, :], numbins=100)
    
        fig, axs = plt.subplots(2, 1, figsize=(12, 9))
    
        axs[0].plot(cumfreq_x(cfq_t_F), cfq_t_F.cumcount/N, label="f generates")
        axs[0].plot(cumfreq_x(cfq_t_G), cfq_t_G.cumcount/N, label="g generates")
        axs[0].grid(linestyle='--')
        axs[0].legend(loc=4)
        axs[0].title.set_text('CDF of duration of unemployment')
        axs[0].set(xlabel='time', ylabel='Prob(time)')
    
        axs[1].plot(cumfreq_x(cfq_π_F), cfq_π_F.cumcount/N, label="f generates")
        axs[1].plot(cumfreq_x(cfq_π_G), cfq_π_G.cumcount/N, label="g generates")
        axs[1].grid(linestyle='--')
        axs[1].legend(loc=4)
        axs[1].title.set_text('CDF of π at time worker accepts wage and leaves unemployment')
        axs[1].set(xlabel='π', ylabel='Prob(π)')
    
        plt.show()

Examples
--------

Example 1 (Baseline)
~~~~~~~~~~~~~~~~~~~~

:math:`F` ~ Beta(1, 1), :math:`G` ~ Beta(3, 1.2), :math:`c`\ =0.3.

The red arrows in the upper right figure show how :math:`\pi_t` is going
to be updated by the new information :math:`w_t`. As the formula above
implies, the direction is determined by the relationship between
:math:`l(w_t)` and :math:`1`.

The magnitude is small if

-  :math:`l(w)` is close to :math:`1`, which means the new :math:`w` is
   not very informative for distinguishing two distributions,
-  :math:`\pi_{t-1}` is close to either :math:`0` or :math:`1`, which
   means the priori is strong.

One question of interest is whether worker will get employed earlier or
not, when the actual ruling distribution is :math:`g` instead of
:math:`f`? The argument has two aspects that go in the oppsite
directions.

-  if f generates, then w is more likely to be low, but we also expect
   :math:`\pi` to move to 1 and lower the threshold for getting employed
   (worker being less selective),
-  if g generates, then w is more likely to be high, but we also expect
   :math:`\pi` to move to 0 and increase the threshold for getting
   employed (worker being more selective).

Quantitatively, the lower right figure sheds lights on which part of the
argument is dominant in this example. It shows the probability of worker
accepting an offer at different π, when :math:`f` or :math:`g` generates
the wage offer. As it implies, under the current parameterization,
worker is always more likely to accept an offer even if the worker
believes the true distribution is :math:`g` and therefore is relatively
more selective. The empirical cumulative distribution of the duration of
unemployment verifies our conjecture.

.. code-block:: python3

    job_search_example()

Example 2
~~~~~~~~~

:math:`F` ~ Beta(1, 1), :math:`G` ~ Beta(1.2, 1.2), :math:`c`\ =0.3.

Now :math:`G` has the same mean as :math:`F` with a smaller variance.
Since the unemployment compensation :math:`c` serves as a lower bound
for bad wage offers, :math:`G` is now an “inferior” distribution to
:math:`F`. Consequently, we observe that the optimal policy
:math:`\overline{w}(\pi)` is increasing in :math:`\pi`.

.. code-block:: python3

    job_search_example(1, 1, 1.2, 1.2, 0.3)

Example 3
~~~~~~~~~

:math:`F` ~ Beta(1, 1), :math:`G` ~ Beta(2, 2), :math:`c`\ =0.3.

If the variance of :math:`G` is smaller, we observe in the result that
:math:`G` is even more “inferior” and the slope of
:math:`\overline{w}(\pi)` is larger.

.. code-block:: python3

    job_search_example(1, 1, 2, 2, 0.3)

Example 4
~~~~~~~~~

:math:`F` ~ Beta(1, 1), :math:`G` ~ Beta(3, 1.2), and :math:`c`\ =0.8.

In this example, we keep the parameters of beta distributions to be the
same with the baseline case, but increase the unemployment compensation
:math:`c`. Comparing to the baseline case (example 1) where the
unemployment compensation if low (:math:`c`\ =0.3), now the worker can
afford a longer learning period. As a result, the worker tends to accept
wage offers much later, and at the time of employment, the belief
:math:`\pi` is either more close to :math:`0` or :math:`1`, which means
the worker has a more clear idea about what the true distribution is
when chooses to accept the wage offer.

.. code-block:: python3

    job_search_example(1, 1, 3, 1.2, c=0.8)

Example 5
~~~~~~~~~

:math:`F` ~ Beta(1, 1), :math:`G` ~ Beta(3, 1.2), and :math:`c`\ =0.1.

As expected, a smaller :math:`c` makes people accept wage offers earlier
with little knowledge accumulated about the true distribution, because
of the more painful unemployment.

.. code-block:: python3

    job_search_example(1, 1, 3, 1.2, c=0.1)

