.. _odu_v3:

.. include:: /_static/includes/header.raw

.. highlight:: python3


Exchangeability and Bayesian Updating
======================================

In addition to what's in Anaconda, this lecture employs the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon
  !pip install interpolation

  from numba import njit, prange, vectorize
  from interpolation import mlinterp, interp
  from math import gamma
  import numpy as np
  import matplotlib.pyplot as plt
  %matplotlib inline
  from matplotlib import cm

.. code-block:: python3

    import scipy.optimize as op
    from scipy.stats import cumfreq, beta

Exchangeable but not i.i.d.
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`\{W_t\}_{t=0}^\infty` be a sequence of nonnegative
scalar random variables with a joint probability distribution
constructed as follows.

There are two distinct cumulative distribution functions :math:`F` and :math:`G`
— with densities :math:`f` and :math:`g` for a nonnegative scalar random
variable :math:`W`.

Before the start of time, say at time :math:`t-1`, “nature” once and for
all selects **either** :math:`f` **or** :math:`g` — and thereafter at each time
:math:`t \geq 0` draws a random :math:`W` from the selected
distribution.

In particular, assume that nature selects :math:`F` with probability
:math:`\tilde \pi \in (0,1)` and
:math:`G` with probability :math:`1 - \tilde \pi`.

Conditional on nature selecting :math:`F`, the joint density of the
sequence :math:`W_0, W_1, \ldots` is

.. math::  f(W_0) f(W_1) \cdots 

Conditional on nature selecting :math:`G`, the joint density of the
sequence :math:`W_0, W_1, \ldots` is

.. math::  g(W_0) g(W_1) \cdots 

Notice that **conditional on nature having selected** :math:`F`, the
sequence :math:`W_0, W_1, \ldots` is independently and
identically distributed; and that **conditional on nature having
selected** :math:`G`, the sequence :math:`W_0, W_1, \ldots` is
independently and identically distributed.

The unconditional distribution of :math:`W_0, W_1, \ldots` is
evidently

.. math::  h(W_0, W_1, \ldots ) \equiv \tilde \pi [f(W_0) f(W_1) \cdots ] + ( 1- \tilde \pi) [g(W_0) g(W_1) \cdots ] 

Under the unconditional distribution :math:`h(W_0, W_1, \ldots )`, the
sequence :math:`W_0, W_1, \ldots` is **not** independently and
identically distributed.

To verify this claim, notice, for example, that

.. math::

    h(w_0, w_1) = \tilde \pi f(w_0)f (w_1) + (1 - \tilde \pi) g(w_0)g(w_1) \neq
                  (\tilde \pi f(w_0) + (1-\tilde \pi) g(w_0))(
                   \tilde \pi f(w_1) + (1-\tilde \pi) g(w_1))  

Thus, the conditional distribution

.. math::

    h(w_1 | w_0) \equiv \frac{h(w_0, w_1)}{(\tilde \pi f(w_0) + (1-\tilde \pi) g(w_0))}
     \neq ( \tilde \pi f(w_1) + (1-\tilde \pi) g(w_1)) 

While the sequence :math:`W_0, W_1, \ldots` is not i.i.d., it is
**exchangeable**, which means that

.. math::  h(w_0, w_1) = h(w_1, w_0) 

and so on.

Let :math:`q` represent the distribution that nature ends up drawing
:math:`w` from and let

.. math::  \tilde \pi = \mathbb{P}\{q = f \} 

Suppose that at :math:`t \geq 0`, we observe a history
:math:`w^t \equiv [w_t, w_{t-1}, \ldots, w_0]`.

Let

.. math::  \pi_t  = \mathbb{P}\{q = f  | w^t \} 

The distribution of :math:`w_{t+1}` conditional on :math:`w^t` is then

.. math::  \pi_t f + (1 - \pi_t) g . 

Bayes’ rule for updating :math:`\pi_{t+1}` is

.. math::


   \pi_{t+1}
   = \frac{\pi_t f(w_{t+1})}{\pi_t f(w_{t+1}) + (1 - \pi_t) g(w_{t+1})}

As another reminder, the last expression follows from Bayes’ rule, which
tells us that

.. math::


   \mathbb{P}\{q = f \,|\, W = w\}
   = \frac{\mathbb{P}\{W = w \,|\, q = f\}\mathbb{P}\{q = f\}}
   {\mathbb{P}\{W = w\}}
   \quad \text{and} \quad
   \mathbb{P}\{W = w\} = \sum_{\omega \in \{f, g\}} \mathbb{P}\{W = w \,|\, q = \omega\} \mathbb{P}\{q = \omega\}

Define the **likelihood ratio** 

.. math::

   l(w) = \frac{f(w)}{g(w)}


It is convenient for us to rewrite the updating rule :eq:`odu_pi_rec` as

.. math::


   \pi_{t+1}   =\frac{\pi_{t}f\left(w_{t+1}\right)}{\pi_{t}f\left(w_{t+1}\right)+\left(1-\pi_{t}\right)g\left(w_{t+1}\right)}
       =\frac{\pi_{t}\frac{f\left(w_{t+1}\right)}{g\left(w_{t+1}\right)}}{\pi_{t}\frac{f\left(w_{t+1}\right)}{g\left(w_{t+1}\right)}+\left(1-\pi_{t}\right)}
       =\frac{\pi_{t}l\left(w_{t+1}\right)}{\pi_{t}l\left(w_{t+1}\right)+\left(1-\pi_{t}\right)}

This implies that

.. math::


   \frac{\pi_{t+1}}{\pi_{t}}=\frac{l\left(w_{t+1}\right)}{\pi_{t}l\left(w_{t+1}\right)+\left(1-\pi_{t}\right)}\begin{cases}
   >1 & \text{if }l\left(w_{t+1}\right)>1\\
   \leq1 & \text{if }l\left(w_{t+1}\right)\leq1
   \end{cases}

We’ll plot :math:`l\left(w\right)` as a way to enlighten us about how
learning – i.e., Bayesian updating of the probability :math:`\pi` that
nature has chosen distribution :math:`f` – works.


Below we define a wrapper function that displays informative graphs
given parameters of :math:`f` and :math:`g`.

.. code-block:: python3

    def learning_example(F_a=1, F_b=1, G_a=3, G_b=1.2):
        """
        Given the parameters that specify F and G distributions,
        display the updating rule of belief π.
        """
    
        f = njit(lambda x: p(x, F_a, F_b))
        g = njit(lambda x: p(x, G_a, G_b))
    
        # l(w) = f(w) / g(w)
        l = lambda w: f(w) / g(w)
        # objective function for solving l(w) = 1
        obj = lambda w: l(w) - 1
    
        x_grid = np.linspace(0, 1, 100)
        π_grid = np.linspace(1e-3, 1-1e-3, 100)
        
        w_max = 1
        w_grid = np.linspace(1e-12, w_max-1e-12, 100)
    
        # the mode of beta distribution
        # use this to divide w into two intervals for root finding
        G_mode = (G_a - 1) / (G_a + G_b - 2)
        roots = np.empty(2)
        roots[0] = op.root_scalar(obj, bracket=[1e-10, G_mode]).root
        roots[1] = op.root_scalar(obj, bracket=[G_mode, 1-1e-10]).root
    
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
        ax1.plot(l(w_grid), w_grid, label='$l$', lw=2)
        ax1.vlines(1., 0., 1., linestyle="--")
        ax1.hlines(roots, 0., 2., linestyle="--")
        ax1.set_xlim([0., 2.])
        ax1.legend(loc=4)
        ax1.set(xlabel='$l(w)=f(w)/g(w)$', ylabel='$w$')
    
        ax2.plot(f(x_grid), x_grid, label='$f$', lw=2)
        ax2.plot(g(x_grid), x_grid, label='$g$', lw=2)
        ax2.vlines(1., 0., 1., linestyle="--")
        ax2.hlines(roots, 0., 2., linestyle="--")
        ax2.legend(loc=4)
        ax2.set(xlabel='$f(w), g(w)$', ylabel='$w$')
    
        area1 = integrate.quad(f, 0, roots[0])[0]
        area2 = integrate.quad(g, roots[0], roots[1])[0]
        area3 = integrate.quad(f, roots[1], 1)[0]
    
        ax2.text(np.mean([f(0), f(roots[0])])/2, np.mean([0, roots[0]]), \
                                                         f"{area1: .3g}")
        ax2.fill_between([0, 1], 0, roots[0], color='blue', alpha=0.15)
        ax2.text(np.mean(g(roots))/2, np.mean(roots), f"{area2: .3g}")
        w_roots = np.linspace(roots[0], roots[1], 20)
        ax2.fill_betweenx(w_roots, 0, g(w_roots), color='orange', alpha=0.15)
        ax2.text(np.mean([f(roots[1]), f(1)])/2, np.mean([roots[1], 1]), \
                                                        f"{area3: .3g}")
        ax2.fill_between([0, 1], roots[1], 1, color='blue', alpha=0.15)
    
        W = np.arange(0.01, 0.99, 0.08)
        Π = np.arange(0.01, 0.99, 0.08)
    
        ΔW = np.zeros((len(W), len(Π)))
        ΔΠ = np.empty((len(W), len(Π)))
        for i, w in enumerate(W):
            for j, π in enumerate(Π):
                lw = l(w)
                ΔΠ[i, j] = π * (lw / (π * lw + 1 - π) - 1)
    
        q = ax3.quiver(Π, W, ΔΠ, ΔW, scale=2, color='r', alpha=0.8)
    
        ax3.fill_between(π_grid, 0, roots[0], color='blue', alpha=0.15)
        ax3.fill_between(π_grid, roots[0], roots[1], color='green', alpha=0.15)
        ax3.fill_between(π_grid, roots[1], w_max, color='blue', alpha=0.15)
        ax3.hlines(roots, 0., 1., linestyle="--")
        ax3.set(xlabel='$\pi$', ylabel='$w$')
        ax3.grid()
    
        plt.show()

.. code-block:: python3

    learning_example()

.. code-block:: python3

    learning_example(G_a=2, G_b=1.6)




TOM EDIT MORE BELOW 
---------------------

.. code-block:: python3

    #= define general beta distribution =#
    @vectorize
    def p(x, a, b):
        r = gamma(a + b) / (gamma(a) * gamma(b))
        return r * x**(a-1) * (1 - x)**(b-1)

.. code-block:: python3

    #= redefine functions for updating and simulation =#
    #= only for discussion about learning                    =#
    
    def function_factory(F_a=1, F_b=1, G_a=3, G_b=1.2):
    
        # define f and g
        f = njit(lambda x: p(x, F_a, F_b))
        g = njit(lambda x: p(x, G_a, G_b))
    
        @njit
        def update(a, b, π):
            "Update π by drawing from beta distribution with parameters a and b"
    
            w = np.random.beta(a, b)       # Draw realization
            π = 1 / (1 + ((1 - π) * g(w)) / (π * f(w)))
    
            return π
    
        @njit
        def simulate_path(a, b, T=600):
            "Simulates path of beliefs π"
    
            π = np.empty(T+1)
    
            # initial condition
            π[0] = 0.5
    
            for t in range(1, T+1):
                π[t] = update(a, b, π[t-1])
    
            return π
    
        def simulate(a=1, b=1, T=50, N=200, display=True):
    
            fig = plt.figure()
            π_paths = np.empty((N, T+1))
    
            if display:
                for i in range(N):
                    π_paths[i] = simulate_path(a=a, b=b, T=T)
                    plt.plot(range(T+1), π_paths[i], color='b', lw=0.8, alpha=0.5)
    
                plt.show()
    
            return π_paths
        
        return simulate

.. code-block:: python3

    simulate = function_factory()

Here we show :math:`N` simulated :math:`\pi_t` paths with :math:`T`
periods.

.. code-block:: python3

    T = 50

.. code-block:: python3

    # when nature selects F
    π_paths_F = simulate(a=1, b=1, T=T, N=1000)

.. code-block:: python3

    # when nature selects G
    π_paths_G = simulate(a=3, b=1.2, T=T, N=1000)

In the following, we compare the convergences of :math:`\pi_t` when the
nature selects :math:`f` or :math:`g`.

Using the simulated :math:`N` :math:`\pi_t` paths, we compute
:math:`1 - \sum_{i=1}^{N}\pi_{i,t}` at each :math:`t` when :math:`f`
generates, and :math:`\sum_{i=1}^{N}\pi_{i,t}` when :math:`g` generates.

.. code-block:: python3

    plt.plot(range(T+1), 1 - np.mean(π_paths_F, 0), label='F generates')
    plt.plot(range(T+1), np.mean(π_paths_G, 0), label='G generates')
    plt.legend()
    plt.title("convergence")

Now we compute the following conditional expectations,

.. math::


   \begin{aligned}
   E\left[\frac{\pi_{t+1}}{\pi_{t}}\biggm|q=\omega, \pi_{t}\right] &=E\left[\frac{l\left(w_{t+1}\right)}{\pi_{t}l\left(w_{t+1}\right)+\left(1-\pi_{t}\right)}\biggm|q=\omega, \pi_{t}\right], \\
       &=\int_{0}^{1}\frac{l\left(w_{t+1}\right)}{\pi_{t}l\left(w_{t+1}\right)+\left(1-\pi_{t}\right)}\omega\left(w_{t+1}\right)dw_{t+1}
   \end{aligned}

where :math:`\omega=f,g`.

.. code-block:: python3

    def expected_ratio(F_a=1, F_b=1, G_a=3, G_b=1.2):
    
        # define f and g
        f = njit(lambda x: p(x, F_a, F_b))
        g = njit(lambda x: p(x, G_a, G_b))
    
        l = lambda w: f(w) / g(w)
        integrand_f = lambda w, π: f(w) * l(w)  / (π * l(w) + 1 - π)
        integrand_g = lambda w, π: g(w) * l(w) / (π * l(w) + 1 - π)
    
        π_grid = np.linspace(0.02, 0.98, 100)
    
        plt.plot(π_grid, [integrate.quad(integrand_f, 0, 1, args=(π,))[0] \
                for π in π_grid], label="f generates")
        plt.plot(π_grid, [integrate.quad(integrand_g, 0, 1, args=(π,))[0] \
                for π in π_grid], label="g generates")
        plt.hlines(1, 0, 1, linestyle="--")
        plt.xlabel("$π_t$")
        plt.ylabel("$E[\pi_{t+1}/\pi_t]$")
        plt.legend()
    
        plt.show()

First, consider the case where :math:`F_a=F_b=1` and
:math:`G_a=3, G_b=1.2`.

.. code-block:: python3

    expected_ratio()

Here shows the case where :math:`f` and :math:`g` are identical beta
distributions, and :math:`F_a=G_a=3, F_b=G_b=1.2`. (The case where there
is nothing to learn.)

.. code-block:: python3

    expected_ratio(F_a=3, F_b=1.2)

Lastly, we show the case where :math:`f` and :math:`g` are neither very
different, and nor identical. :math:`F_a=2, F_b=1` and
:math:`G_a=3, G_b=1.2`.

.. code-block:: python3

    expected_ratio(F_a=2, F_b=1, G_a=3, G_b=1.2)
