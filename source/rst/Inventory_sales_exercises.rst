.. _information_sales_exercises:

.. include:: /_static/includes/header.raw

.. highlight:: python3


Exercises for Production Smoothing via Inventories
--------------------------------------------------

.. contents:: :depth: 2

**Co-authors: Thomas J. Sargent**


In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon


Here is code for computing an optimal decision rule and for analyzing
its consequences

.. code-block:: ipython

    import numpy as np
    import quantecon as qe
    import matplotlib.pyplot as plt
    %matplotlib inline

.. code-block:: python3

    class smoothing_example:
        """
        Class for constructing, solving, and plotting results for
        an inventories and sales smoothing problem.
        """
    
        def __init__(self,
                     β=0.96,           # discount factor
                     c1=1,             # cost-of-production
                     c2=1,
                     d1=1,             # cost-of-holding inventories
                     d2=1,
                     a0=10,            # inverse demand function
                     a1=1,
                     A22=[[1,   0],    # z process
                          [1, 0.9]],
                     C2=[[0], [1]],
                     G=[0, 1]):
    
            self.β = β
            self.c1, self.c2 = c1, c2
            self.d1, self.d2 = d1, d2
            self.a0, self.a1 = a0, a1
            self.A22 = np.atleast_2d(A22)
            self.C2 = np.atleast_2d(C2)
            self.G = np.atleast_2d(G)
    
            # dimensions
            k, j = self.C2.shape        # dimensions for randomness part
            n = k + 1                   # number of states
            m = 2                       # number of controls
            
            Sc = np.zeros(k)
            Sc[0] = 1
    
            # construct matrices of transition law
            A = np.zeros((n, n))
            A[0, 0] = 1
            A[1:, 1:] = A22
    
            B = np.zeros((n, m))
            B[0, :] = 1, -1
    
            C = np.zeros((n, j))
            C[1:, :] = C2
    
            self.A, self.B, self.C = A, B, C
    
            # construct matrices of one period profit function
            R = np.zeros((n, n))
            R[0, 0] = d2
            R[1:, 0] = d1 / 2 * Sc
            R[0, 1:] = d1 / 2 * Sc
    
            Q = np.zeros((m, m))
            Q[0, 0] = c2
            Q[1, 1] = a1 + d2
    
            N = np.zeros((m, n))
            N[1, 0] = - d2
            N[0, 1:] = c1 / 2 * Sc
            N[1, 1:] = - a0 / 2 * Sc - self.G / 2
    
            self.R, self.Q, self.N = R, Q, N
    
            # construct LQ instance
            self.LQ = qe.LQ(Q, R, A, B, C, N, beta=β)
            self.LQ.stationary_values()
    
        def simulate(self, x0, T=100):
    
            c1, c2 = self.c1, self.c2
            d1, d2 = self.d1, self.d2
            a0, a1 = self.a0, self.a1
            G = self.G
    
            x_path, u_path, w_path = self.LQ.compute_sequence(x0, ts_length=T)
    
            I_path = x_path[0, :-1]
            z_path = x_path[1:, :-1]
            𝜈_path = (G @ z_path)[0, :]
    
            Q_path = u_path[0, :]
            S_path = u_path[1, :]
    
            revenue = (a0 - a1 * S_path + 𝜈_path) * S_path
            cost_production = c1 * Q_path + c2 * Q_path ** 2
            cost_inventories = d1 * I_path + d2 * (S_path - I_path) ** 2
    
            Q_no_inventory = (a0 + 𝜈_path - c1) / (2 * (a1 + c2))
            Q_hardwired = (a0 + 𝜈_path - c1) / (2 * (a1 + c2 + d2))
    
            fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    
            ax[0, 0].plot(range(T), I_path, label="inventories")
            ax[0, 0].plot(range(T), S_path, label="sales")
            ax[0, 0].plot(range(T), Q_path, label="production")
            ax[0, 0].legend(loc=1)
            ax[0, 0].set_title("inventories, sales, and production")
    
            ax[0, 1].plot(range(T), (Q_path - S_path), color='b')
            ax[0, 1].set_ylabel("change in inventories", color='b')
            span = max(abs(Q_path - S_path))
            ax[0, 1].set_ylim(0-span*1.1, 0+span*1.1)
            ax[0, 1].set_title("demand shock and change in inventories")
    
            ax1_ = ax[0, 1].twinx()
            ax1_.plot(range(T), 𝜈_path, color='r')
            ax1_.set_ylabel("demand shock", color='r')
            span = max(abs(𝜈_path))
            ax1_.set_ylim(0-span*1.1, 0+span*1.1)
    
            ax1_.plot([0, T], [0, 0], '--', color='k')
    
            ax[1, 0].plot(range(T), revenue, label="revenue")
            ax[1, 0].plot(range(T), cost_production, label="cost_production")
            ax[1, 0].plot(range(T), cost_inventories, label="cost_inventories")
            ax[1, 0].legend(loc=1)
            ax[1, 0].set_title("profits decomposition")
    
            ax[1, 1].plot(range(T), Q_path, label="production")
            ax[1, 1].plot(range(T), Q_hardwired, label='production when  $I_t$ forced to be zero')
            ax[1, 1].plot(range(T), Q_no_inventory, label='production when inventories not useful')
            ax[1, 1].legend(loc=1)
            ax[1, 1].set_title('three production concepts')
    
            plt.show()

Exercises
---------

Please try to analyze some inventory sales smoothing problems using the
``smoothing_example`` class.

Exercise 1
~~~~~~~~~~

Assume the demand shock follows AR(2) process below:

.. math::


   \nu_{t}=\alpha+\rho_{1}\nu_{t-1}+\rho_{2}\nu_{t-2}+\epsilon_{t}.

You need to construct :math:`A22`, :math:`C`, and :math:`G` matrices
properly, and then input them as the keyword arguments of
``smoothing_example`` class. Simulate paths starting from the initial
condition :math:`x_0 = \left[0, 1, 0, 0\right]^\prime`.

After this, try to construct a very similar ``smoothing_example`` with
the same demand shock process but exclude the randomness
:math:`\epsilon_t`. Compute the stationary states :math:`\bar{x}` by
simulating for a long period. Then try to add shocks with different
magnitude to :math:`\bar{\nu}_t` and simulate paths. You should see how
firms respond differently by staring at the production plans.

Exercise 2
~~~~~~~~~~

Change parameters of :math:`C(Q_t)` and :math:`d(I_t, S_t)`.

1. Make production more costly, by setting :math:`c_2=5`.
2. Increase the cost of having inventories deviate from sales, by
   setting :math:`d_2=5`.

Solution 1
~~~~~~~~~~

.. code-block:: python3

    # set parameters
    α = 1
    ρ1 = 1.2
    ρ2 = -.3

.. code-block:: python3

    # construct matrices
    A22 =[[1,  0,  0],
              [1, ρ1, ρ2],
              [0,  1, 0]]
    C2 = [[0], [1], [0]]
    G = [0, 1, 0]

.. code-block:: python3

    ex1 = smoothing_example(A22=A22, C2=C2, G=G)
    
    x0 = [0, 1, 0, 0] # initial condition
    ex1.simulate(x0)

.. code-block:: python3

    # now silence the noise
    ex1_no_noise = smoothing_example(A22=A22, C2=[[0], [0], [0]], G=G)
    
    # initial condition
    x0 = [0, 1, 0, 0]
    
    # compute stationary states
    x_bar = ex1_no_noise.LQ.compute_sequence(x0, ts_length=250)[0][:, -1]
    x_bar

In the following, we add small and large shocks to :math:`\bar{\nu}_t`
and compare how firm responds differently in quantity. As the shock is
not very persistent under the parameterization we are using, we focus on
a short period response.

.. code-block:: python3

    T = 40

.. code-block:: python3

    # small shock
    x_bar1 = x_bar.copy()
    x_bar1[2] += 2
    ex1_no_noise.simulate(x_bar1, T=T)

.. code-block:: python3

    # large shock
    x_bar1 = x_bar.copy()
    x_bar1[2] += 10
    ex1_no_noise.simulate(x_bar1, T=T)

Solution 2
~~~~~~~~~~

.. code-block:: python3

    x0 = [0, 1, 0]

.. code-block:: python3

    smoothing_example(c2=5).simulate(x0)

.. code-block:: python3

    smoothing_example(d2=5).simulate(x0)

