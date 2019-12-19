
.. code-block:: ipython

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline
    from numba import vectorize, njit
    from math import gamma

Assume that nature chooses :math:`q` as the true random process that
generates data, and :math:`q=f,g`.

Define the likelihood ratio process with sequence
:math:`\left\{ l\left(w_{t}\right)\right\} _{t=0}^{\infty}` as

.. math::


   L\left(w^{t}\right)=\prod_{i=0}^{t}l\left(w_{i}\right).

where

.. math::


   l\left(w_{t}\right)=\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)},\quad t\geq0.

and :math:`w^{t}=\left\{ w_{0},w_{1},\dots,w_{t}\right\}` is the
history.

Notice that

.. math::


   L\left(w^t\right) = L\left(w^{t-1}\right) l\left(w_t\right).

Here we define functions of beta distributions and simulation of
likelihood ratio paths.

.. code-block:: python3

    F_a, F_b = 1, 1
    G_a, G_b = 3, 1.2
    
    @vectorize
    def p(x, a, b):
        r = gamma(a + b) / (gamma(a) * gamma(b))
        return r * x** (a-1) * (1 - x) ** (b-1)
    
    f = njit(lambda x: p(x, F_a, F_b))
    g = njit(lambda x: p(x, G_a, G_b))

.. code-block:: python3

    @njit
    def simulate(a, b, T=30, N=5000):
    
        l_arr = np.empty((N, T))
    
        for i in range(N):
    
            for j in range(T):
                w = np.random.beta(a, b)
                l_arr[i, j] = f(w) / g(w)
    
        return l_arr

Consider first the nature selects :math:`g`.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below shows the simulation results.

.. code-block:: python3

    l_arr = simulate(G_a, G_b)
    l_seq = np.cumprod(l_arr, axis=1)

.. code-block:: python3

    N, T = l_arr.shape
    
    for i in range(N):
        
        plt.plot(range(T), l_seq[i, :], color='b', lw=0.8, alpha=0.5)
        
    plt.ylim([0, 3])
    plt.title("$L(w^{t})$ paths");

We see that most of the probability mass shifts toward zero as :math:`T`
grows.

To see it clearly, we plot the probability mass of
:math:`L\left(w^{t}\right)` falling into the interval
:math:`\left[0, 0.01\right]` over time.

.. code-block:: python3

    plt.plot(range(T), np.sum(l_seq <= 0.01, axis=0) / N)

However, one peculiar fact is that the unconditional mean of
:math:`L\left(w^t\right)` is :math:`1` for all :math:`t`.

To see why, first notice that the unconditional mean
:math:`E_{0}\left[l\left(w_{t}\right)\bigm|q=g\right]` is :math:`1` for
all :math:`t`:

.. math::


   \begin{aligned}
   E_{0}\left[l\left(w_{t}\right)\bigm|q=g\right]  &=\int\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}g\left(w_{t}\right)dw_{t} \\
       &=\int f\left(w_{t}\right)dw_{t} \\
       &=1,
   \end{aligned}

which implies immediately

.. math::


   \begin{aligned}
   E_{0}\left[L\left(w^{0}\right)\bigm|q=g\right]  &=E_{0}\left[l\left(w_{0}\right)\bigm|q=g\right]\\
       &=1.\\
   \end{aligned}

Because :math:`L(w^t)` is multiplicative and :math:`\{w_t\}_{t=0}^t` is
i.i.d. sequence, we therefore have

.. math::


   \begin{aligned}
   E_{0}\left[L\left(w^{t}\right)\bigm|q=g\right]  &=E_{0}\left[L\left(w^{t-1}\right)l\left(w_{t}\right)\bigm|q=g\right] \\
       &=E_{0}\left[L\left(w^{t-1}\right)E\left[l\left(w_{t}\right)\bigm|q=g,w^{t-1}\right]\bigm|q=g\right] \\
       &=E_{0}\left[L\left(w^{t-1}\right)E\left[l\left(w_{t}\right)\bigm|q=g\right]\bigm|q=g\right] \\
       &=E_{0}\left[L\left(w^{t-1}\right)\bigm|q=g\right] \\
   \end{aligned}

for any :math:`t \geq 1`.

Mathematical induction implies
:math:`E_{0}\left[L\left(w^{t}\right)\bigm|q=g\right]=1` for all
:math:`t \geq 0`.

To verify this, we simulate larger sample with longer period and
calculate the means of :math:`L\left(w^t\right)` at each :math:`t`.

.. code-block:: python3

    l_arr = simulate(G_a, G_b, T=100, N=50000)
    l_seq = np.cumprod(l_arr, axis=1)

Compute the unconditional mean
:math:`E_{0}\left[L\left(w^{t}\right)\right]`.

.. code-block:: python3

    N, T = l_arr.shape
    plt.plot(range(T), np.mean(l_arr, axis=0))
    plt.hlines(1, 0, T, linestyle='--')

Consider now the nature selects :math:`f`.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case, the unconditional mean explodes very quickly, because

.. math::


   \begin{aligned}
   E_{0}\left[l\left(w_{t}\right)\bigm|q=f\right]  &=\int\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}f\left(w_{t}\right)dw_{t} \\
       &=\int\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}g\left(w_{t}\right)dw_{t} \\
       &=\int l\left(w_{t}\right)^{2}g\left(w_{t}\right)dw_{t} \\
       &=E_{0}\left[l\left(w_{t}\right)^{2}\mid q=g\right] \\
       &=E_{0}\left[l\left(w_{t}\right)\mid q=g\right]^{2}+Var\left(l\left(w_{t}\right)\mid q=g\right) \\
       &>E_{0}\left[l\left(w_{t}\right)\mid q=g\right]^{2} \\
       &=1 \\
   \end{aligned}

The simulation result below confirms this conclusion. Please note the
scale of y axis.

.. code-block:: python3

    l_arr = simulate(F_a, F_b, T=30, N=50000)
    l_seq = np.cumprod(l_arr, axis=1)

.. code-block:: python3

    N, T = l_arr.shape
    plt.plot(range(T), np.mean(l_seq, axis=0))

We also plot the probability mass of :math:`L\left(w^t\right)` falling
into the interval :math:`[10000, \infty)` over time, and see how fast
the probability mass moves towards :math:`\infty`.

.. code-block:: python3

    plt.plot(range(T), np.sum(l_seq > 10000, axis=0) / N)

