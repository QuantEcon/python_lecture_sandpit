.. _cass_koopmans_1:

.. include:: /_static/includes/header.raw

.. highlight:: python3


*************************************
Cass-Koopmans Planning Problem
*************************************

.. contents:: :depth: 2

Overview
=========

This lecture and the following one XXXX describe a model that Tjalling Koopmans :cite:`Koopmans`
and David Cass :cite:`Cass` used to analyze optimal growth.

The model can be viewed as an extension of the model of Robert Solow
described in `an earlier lecture <https://lectures.quantecon.org/py/python_oop.html>`__
but adapted to make the savings rate the outcome of an optimal choice.

(Solow assumed a constant saving rate determined outside the model).


We describe two versions of the model, one in this lecture and the other in XXXX

Together, the two lectures  illustrate what is, in fact, a
more general connection between a **planned economy** and an economy
organized as a **competitive equilibrium**.

This lecture is devoted to the planned economy version.  

The lecture uses important ideas including

-  A min-max problem for solving a planning problem.

-  A **shooting algorithm** for solving difference equations subject
   to initial and terminal conditions.

-  A **turnpike** property that describes optimal paths for
   long-but-finite horizon economies.

Let's start with some standard imports:

.. code-block:: ipython

  from numba import njit
  import numpy as np
  import matplotlib.pyplot as plt
  %matplotlib inline

The Growth Model
==================

Time is discrete and takes values :math:`t = 0, 1 , \ldots, T` where :math:`T` is  finite.

(We'll study a limiting case in which  :math:`T = + \infty` before concluding).

A single good can either be consumed or invested in physical capital.

The consumption good is not durable and depreciates completely if not
consumed immediately.

The capital good is durable but depreciates some each period.

We let :math:`C_t` be a nondurable consumption good at time :math:`t`.

Let :math:`K_t` be the stock of physical capital at time :math:`t`.

Let :math:`\vec{C}` = :math:`\{C_0,\dots, C_T\}` and
:math:`\vec{K}` = :math:`\{K_1,\dots,K_{T+1}\}`.

A representative household is endowed with one unit of labor at each
:math:`t` and likes the consumption good at each :math:`t`.

The representative household inelastically supplies a single unit of
labor :math:`N_t` at each :math:`t`, so that
:math:`N_t =1 \text{ for all } t \in [0,T]`.

The representative household has preferences over consumption bundles
ordered by the utility functional:

.. math::
    :label: utility-functional

    U(\vec{C}) = \sum_{t=0}^{T} \beta^t \frac{C_t^{1-\gamma}}{1-\gamma}

where :math:`\beta \in (0,1)` is a discount factor and :math:`\gamma >0`
governs the curvature of the one-period utility function with larger :math:`\gamma` implying more curvature.

Note that

.. math::
    :label: utility-oneperiod

    u(C_t) = \frac{C_t^{1-\gamma}}{1-\gamma}

satisfies :math:`u'>0,u''<0`.

:math:`u' > 0` asserts that the consumer prefers more to less.

:math:`u''< 0` asserts that marginal utility declines with increases
in :math:`C_t`.

We assume that :math:`K_0 > 0` is an  exogenous  initial
capital stock.

There is an economy-wide production function

.. math::
  :label: production-function

  F(K_t,N_t) = A K_t^{\alpha}N_t^{1-\alpha}

with :math:`0 < \alpha<1`, :math:`A > 0`.

A feasible allocation :math:`\vec C, \vec K` satisfies

.. math::
  :label: allocation

  C_t + K_{t+1} \leq F(K_t,N_t) + (1-\delta) K_t, \quad \text{for all } t \in [0, T]

where :math:`\delta \in (0,1)` is a depreciation rate of capital.

Planning Problem
------------------

A planner chooses an allocation :math:`\{\vec{C},\vec{K}\}` to
maximize :eq:`utility-functional` subject to :eq:`allocation`.

Let :math:`\vec{\mu}=\{\mu_0,\dots,\mu_T\}` be a sequence of
nonnegative **Lagrange multipliers**.

To find an optimal allocation, form a Lagrangian

.. math::

  \mathcal{L}(\vec{C},\vec{K},\vec{\mu}) =
  \sum_{t=0}^T \beta^t\left\{ u(C_t)+ \mu_t
  \left(F(K_t,1) + (1-\delta) K_t- C_t - K_{t+1} \right)\right\}

and then pose the following min-max problem:

.. math::
  :label: min-max-prob

  \min_{\vec{\mu}} \max_{\vec{C},\vec{K}} \mathcal{L}(\vec{C},\vec{K},\vec{\mu})

Useful Properties of Linearly Homogeneous Production Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following technicalities will help us.

Notice that

.. math::

  F(K_t,N_t) = A K_t^\alpha N_t^{1-\alpha} = N_t A\left(\frac{K_t}{N_t}\right)^\alpha

Define the **output per-capita production function**

.. math::

  \frac{F(K_t,N_t)}{N_t} \equiv f\left(\frac{K_t}{N_t}\right) = A\left(\frac{K_t}{N_t}\right)^\alpha

whose argument is **capital per-capita**.


It is useful to recall the following calculations for the marginal product of capital 


.. math::
  :label: useful-calc1

  \begin{aligned}
  \frac{\partial F(K_t,N_t)}{\partial K_t}
  & =
  \frac{\partial N_t f\left( \frac{K_t}{N_t}\right)}{\partial K_t}
  \\ &=
  N_t f'\left(\frac{K_t}{N_t}\right)\frac{1}{N_t} \quad \text{(Chain rule)}
  \\ &=
  f'\left.\left(\frac{K_t}{N_t}\right)\right|_{N_t=1}
  \\ &= f'(K_t)
  \end{aligned}

and the marginal product of labor

.. math::

  \begin{aligned}
  \frac{\partial F(K_t,N_t)}{\partial N_t}
  &=
  \frac{\partial N_t f\left( \frac{K_t}{N_t}\right)}{\partial N_t} \quad \text{(Product rule)}
  \\ &=
  f\left(\frac{K_t}{N_t}\right){+} N_t f'\left(\frac{K_t}{N_t}\right) \frac{-K_t}{N_t^2} \quad \text{(Chain rule)}
  \\ &=
  f\left(\frac{K_t}{N_t}\right){-}\frac{K_t}{N_t}f'\left.\left(\frac{K_t}{N_t}\right)\right|_{N_t=1}
  \\ &=
  f(K_t) - f'(K_t) K_t
  \end{aligned}


Back to Solving the Problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To solve the Lagrangian extremization problem, we compute first
derivatives of the Lagrangian and set them equal to 0.

-  **Note:** Our problem satisfies
   conditions that assure that required second-order
   conditions are satisfied at an allocation that satisfies the
   first-order conditions that we are about to compute.

-  **Note:**  **Extremization** means
   maximization with respect to :math:`\vec C, \vec K` and
   minimization with respect to :math:`\vec \mu`). 

Here are the **first order necessary conditions** for extremization:


.. math::
    :label: constraint1

    C_t: \qquad u'(C_t)-\mu_t=0 \qquad \text{for all} \quad t= 0,1,\dots,T

.. math::
    :label: constraint2

    K_t: \qquad \beta \mu_t\left[(1-\delta)+f'(K_t)\right] - \mu_{t-1}=0 \qquad \text{for all } \quad t=1,2,\dots,T

.. math::
    :label: constraint3

    \mu_t:\qquad F(K_t,1)+ (1-\delta) K_t  - C_t - K_{t+1}=0 \qquad \text{for all } \quad t=0,1,\dots,T

.. math::
    :label: constraint4

    K_{T+1}: \qquad -\mu_T \leq 0, \ \leq 0 \text{ if } K_{T+1}=0; \ =0 \text{ if } K_{T+1}>0


In computing  :eq:`constraint3` we recognize that 
of :math:`K_t` appears in both the time  :math:`t` and time :math:`t-1`
feasibility constraints.

:eq:`constraint4` comes from differentiating with respect
to :math:`K_{T+1}` and applying the following **Karush-Kuhn-Tucker condition** (KKT)
(see `Karush-Kuhn-Tucker conditions <https://en.wikipedia.org/wiki/Karush-Kuhn-Tucker_conditions>`__):

.. math::
    :label: kkt

    \mu_T K_{T+1}=0



Combining :eq:`constraint1` and :eq:`constraint2` gives

.. math::
  u'\left(C_t\right)\left[(1-\delta)+f'\left(K_t\right)\right]-u'\left(C_{t-1}\right)=0
  \quad \text{ for all } t=1,2,\dots, T+1

which can be rearranged to become

.. math::
    :label: l12

    u'\left(C_{t+1}\right)\left[(1-\delta)+f'\left(K_{t+1}\right)\right]=
    u'\left(C_{t}\right) \quad \text{ for all } t=0,1,\dots, T

Applying  the inverse of the utility function on both sides of the above
equation gives

.. math::
  C_{t+1} =u'^{-1}\left(\left(\frac{\beta}{u'(C_t)}[f'(K_{t+1}) +(1-\delta)]\right)^{-1}\right)

which for our utility function :eq:`utility-oneperiod` becomes the consumption **Euler
equation**

.. math::

  \begin{aligned} C_{t+1} =\left(\beta C_t^{\gamma}[f'(K_{t+1}) +
  (1-\delta)]\right)^{1/\gamma} \notag\\= C_t\left(\beta [f'(K_{t+1}) +
  (1-\delta)]\right)^{1/\gamma} \end{aligned}


We now write Python code for some 
variables and functions that we'll want in order to solve the planning
problem.

.. code-block:: python3

    @njit
    def u(c, γ):
        '''
        Utility function
        ASIDE: If you have a utility function that is hard to solve by hand
        you can use automatic or symbolic  differentiation
        See https://github.com/HIPS/autograd
        '''
        if γ == 1:
            # If γ = 1 we can show via L'hopital's Rule that the utility
            # becomes log
            return np.log(c)
        else:
            return c**(1 - γ) / (1 - γ)

    @njit
    def u_prime(c, γ):
        '''Derivative of utility'''
        if γ == 1:
            return 1 / c
        else:
            return c**(-γ)

    @njit
    def u_prime_inv(c, γ):
        '''Inverse utility'''
        if γ == 1:
            return c
        else:
            return c**(-1 / γ)

    @njit
    def f(A, k, α):
        '''Production function'''
        return A * k**α

    @njit
    def f_prime(A, k, α):
        '''Derivative of production function'''
        return α * A * k**(α - 1)

    @njit
    def f_prime_inv(A, k, α):
        return (k / (A * α))**(1 / (α - 1))


Shooting Method
----------------

We shall use a **shooting method** to compute an optimal allocation
:math:`\vec C, \vec K` and an associated Lagrange multiplier sequence
:math:`\vec \mu`.

The first-order necessary conditions 
:eq:`constraint1`, :eq:`constraint2`, and
:eq:`constraint3`  for the planning problem form a system of **difference equations** with
two boundary conditions:

-  :math:`K_0` is a given **initial condition** for capital

-  :math:`K_{T+1} =0` is a **terminal condition** for capital that we
   deduced from the first-order necessary condition for :math:`K_{T+1}`
   the KKT condition :eq:`kkt`

We have no initial condition for the Lagrange multiplier
:math:`\mu_0`.

If we did, our job would be easy:

-  Given :math:`\mu_0` and :math:`k_0`, we could compute :math:`c_0` from
   equation :eq:`constraint1` and then :math:`k_1` from equation
   :eq:`constraint3` and :math:`\mu_1` from equation
   :eq:`constraint2`.

-  We could continue in this way to compute the remaining elements of
   :math:`\vec C, \vec K, \vec \mu`.

But we don't have an initial condition for :math:`\mu_0`, so this
won't work.

But a simple modification called the **shooting algorithm** 
works.

It is  an instance of a **guess and verify**
algorithm consisting of the following steps:

-  Guess a value for the initial Lagrange multiplier :math:`\mu_0`.

-  Apply the **simple algorithm** described above.

-  Compute the implied value of :math:`k_{T+1}` and check whether it
   equals zero.

-  If the implied :math:`K_{T+1} =0`, we have solved the problem.

-  If :math:`K_{T+1} > 0`, lower :math:`\mu_0` and try again.

-  If :math:`K_{T+1} < 0`, raise :math:`\mu_0` and try again.

The following Python code implements the shooting algorithm for the
planning problem.

We actually modify the algorithm slightly by starting with a guess for 
:math:`c_0` instead of :math:`\mu_0` in the following code. 

We'll start with an incorrect guess.

.. code-block:: python3

    # Parameters
    γ = 2
    δ = 0.02
    β = 0.95
    α = 0.33
    A = 1

    # Initial guesses
    T = 10
    c = np.zeros(T+1)  # T periods of consumption initialized to 0
    # T periods of capital initialized to 0 (T+2 to include t+1 variable as well)
    k = np.zeros(T+2)
    k[0] = 0.3  # Initial k
    c[0] = 0.2  # Guess of c_0

    @njit
    def shooting_method(c, # Initial consumption
                        k,   # Initial capital
                        γ,   # Coefficient of relative risk aversion
                        δ,   # Depreciation rate on capital# Depreciation rate
                        β,   # Discount factor
                        α,   # Return to capital per capita
                        A):  # Technology

        T = len(c) - 1

        for t in range(T):
            # Equation 1 with inequality
            k[t+1] = f(A=A, k=k[t], α=α) + (1 - δ) * k[t] - c[t]
            if k[t+1] < 0:   # Ensure nonnegativity
                k[t+1] = 0

          # Equation 2: We keep in the general form to show how we would
          # solve if we didn't want to do any simplification

            if β * (f_prime(A=A, k=k[t+1], α=α) + (1 - δ)) == np.inf:
                # This only occurs if k[t+1] is 0, in which case, we won't
                # produce anything next period, so consumption will have to be 0
                c[t+1] = 0
            else:
                c[t+1] = u_prime_inv(u_prime(c=c[t], γ=γ) \
                / (β * (f_prime(A=A, k=k[t+1], α=α) + (1 - δ))), γ=γ)

        # Terminal condition calculation
        k[T+1] = f(A=A, k=k[T], α=α) + (1 - δ) * k[T] - c[T]

        return c, k

    paths = shooting_method(c, k, γ, δ, β, α, A)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ['blue', 'red']
    titles = ['Consumption', 'Capital']
    ylabels = ['$c_t$', '$k_t$']

    for path, color, title, y, ax in zip(paths, colors, titles, ylabels, axes):
        ax.plot(path, c=color, alpha=0.7)
        ax.set(title=title, ylabel=y, xlabel='t')

    ax.scatter(T+1, 0, s=80)
    ax.axvline(T+1, color='k', ls='--', lw=1)

    plt.tight_layout()
    plt.show()

Evidently, our initial guess for :math:`\mu_0` is too high and makes
initial consumption too low.

We know this because we miss our :math:`K_{T+1}=0` target on the high
side.

Now we automate things with a search-for-a-good :math:`\mu_0`
algorithm that stops when we hit the target :math:`K_{t+1} = 0`.

The search procedure is to use a **bisection method**.

We make an initial guess for :math:`C_0` (we can eliminate
:math:`\mu_0` because :math:`C_0` is an exact function of
:math:`\mu_0`).

We know that the lowest :math:`C_0` can ever be is :math:`0` and the
largest it can be is initial output :math:`f(K_0)`.

We make a :math:`C_0` guess and shoot forward to :math:`T+1`.

If the :math:`K_{T+1}>0`, we take it to be our new **lower** bound
on :math:`C_0`.

If :math:`K_{T+1}<0`, we take  it to be our new **upper** bound.

Then we make a new guess for :math:`C_0` that is  halfway between our new
upper and lower bounds.

We then shoot forward again, iterating on these steps until we converge.  

When :math:`K_{T+1}` gets close enough to 0 (within some error
tolerance bounds), we stop and declare victory.

.. code-block:: python3

    @njit
    def bisection_method(c,
                         k,
                         γ,              # Coefficient of relative risk aversion
                         δ,              # Depreciation rate
                         β,              # Discount factor
                         α,              # Return to capital per capita
                         A,              # Technology
                         tol=1e-4,
                         max_iter=1e4,
                         terminal=0):    # Value we are shooting towards

        T = len(c) - 1
        i = 1                            # Initial iteration
        c_high = f(k=k[0], α=α, A=A)     # Initial high value of c
        c_low = 0                        # Initial low value of c

        path_c, path_k = shooting_method(c, k, γ, δ, β, α, A)

        while (np.abs((path_k[T+1] - terminal)) > tol or path_k[T] == terminal) \
            and i < max_iter:

            if path_k[T+1] - terminal > tol:
                # If assets are too high the c[0] we chose is now a lower bound
                # on possible values of c[0]
                c_low = c[0]
            elif path_k[T+1] - terminal < -tol:
                # If assets fell too quickly, the c[0] we chose is now an upper
                # bound on possible values of c[0]
                c_high=c[0]
            elif path_k[T] == terminal:
                # If assets fell  too quickly, the c[0] we chose is now an upper
                # bound on possible values of c[0]
                c_high=c[0]

            c[0] = (c_high + c_low) / 2  # This is the bisection part
            path_c, path_k = shooting_method(c, k, γ, δ, β, α, A)
            i += 1

        if np.abs(path_k[T+1] - terminal) < tol and path_k[T] != terminal:
            print('Converged successfully on iteration', i-1)
        else:
            print('Failed to converge and hit maximum iteration')

        μ = u_prime(c=path_c, γ=γ)
        return path_c, path_k, μ

Now we can plot

.. code-block:: python3

    T = 10
    c = np.zeros(T+1) # T periods of consumption initialized to 0
    # T periods of capital initialized to 0. T+2 to include t+1 variable as well
    k = np.zeros(T+2)

    k[0] = 0.3 # initial k
    c[0] = 0.3 # our guess of c_0

    paths = bisection_method(c, k, γ, δ, β, α, A)

    def plot_paths(paths, axes=None, ss=None):

        T = len(paths[0])

        if axes is None:
            fix, axes = plt.subplots(1, 3, figsize=(13, 3))

        ylabels = ['$c_t$', '$k_t$', '$\mu_t$']
        titles = ['Consumption', 'Capital', 'Lagrange Multiplier']

        for path, y, title, ax in zip(paths, ylabels, titles, axes):
            ax.plot(path)
            ax.set(ylabel=y, title=title, xlabel='t')

        # Plot steady state value of capital
        if ss is not None:
            axes[1].axhline(ss, c='k', ls='--', lw=1)

        axes[1].axvline(T, c='k', ls='--', lw=1)
        axes[1].scatter(T, paths[1][-1], s=80)
        plt.tight_layout()

    plot_paths(paths)


Setting Initial Capital to Steady State Capital
----------------------------------------------------

When  :math:`T \rightarrow +\infty`, the optimal allocation converges to
steady state values of :math:`C_t` and :math:`K_t`.

It is instructive to set :math:`K_0` equal
to the :math:`\lim_{T \rightarrow + \infty } K_t`, which we'll call  steady state capital.

In a steady state :math:`K_{t+1} = K_t=\bar{K}` for all very
large :math:`t`.

Evalauating the feasibility constraint :eq:`allocation` at :math \bar K` gives

.. math::
    :label: feasibility-constraint

    f(\bar{K})-\delta \bar{K} = \bar{C}

Substituting :math:`K_t = \bar K` and :math:`C_t=\bar C` for
all :math:`t` into :eq:`l12` gives

.. math:: 1=\beta \frac{u'(\bar{C})}{u'(\bar{C})}[f'(\bar{K})+(1-\delta)]

Defining :math:`\beta = \frac{1}{1+\rho}`, and cancelling gives

.. math:: 1+\rho = 1[f'(\bar{K}) + (1-\delta)]

Simplifying gives

.. math:: f'(\bar{K}) = \rho +\delta

and

.. math:: \bar{K} = f'^{-1}(\rho+\delta)

For the  production function :eq:`production-function` this becomes

.. math:: \alpha \bar{K}^{\alpha-1} = \rho + \delta

As an example, using :math:`\alpha= .33`,
:math:`\rho = 1/\beta-1 =1/(19/20)-1 = 20/19-19/19 = 1/19`, :math:`\delta = 1/50`,
we get

.. math:: \bar{K} = \left(\frac{\frac{33}{100}}{\frac{1}{50}+\frac{1}{19}}\right)^{\frac{67}{100}} \approx 9.57583

Let's verify this with Python and then use this steady state
:math:`\bar K` as our initial capital stock :math:`K_0`.

.. code-block:: python3

    ρ = 1 / β - 1
    k_ss = f_prime_inv(k=ρ+δ, A=A, α=α)

    print(f'steady state for capital is: {k_ss}')

Now we plot

.. code-block:: python3

    T = 150
    c = np.zeros(T+1)
    k = np.zeros(T+2)
    c[0] = 0.3
    k[0] = k_ss  # Start at steady state
    paths = bisection_method(c, k, γ, δ, β, α, A)

    plot_paths(paths, ss=k_ss)

Evidently,  with a large value of
:math:`T`, :math:`K_t` stays near :math:`K_0` until :math:`t` approaches :math:`T` closely.


Let's see what the planner does when we set
:math:`K_0` below :math:`\bar K`.

.. code-block:: python3

    k_init = k_ss / 3   # Below our steady state
    T = 150
    c = np.zeros(T+1)
    k = np.zeros(T+2)
    c[0] = 0.3
    k[0] = k_init
    paths = bisection_method(c, k, γ, δ, β, α, A)

    plot_paths(paths, ss=k_ss)

Notice how the planner pushes capital toward the steady state, stays
near there for a while, then pushes :math:`K_t` toward the terminal
value :math:`K_{T+1} =0` when :math:`t` closely approaches :math:`T`.

The following graphs compare optimal outcomes as we vary :math:`T`.

.. code-block:: python3

    T_list = (150, 75, 50, 25)

    fix, axes = plt.subplots(1, 3, figsize=(13, 3))

    for T in T_list:
        c = np.zeros(T+1)
        k = np.zeros(T+2)
        c[0] = 0.3
        k[0] = k_init
        paths = bisection_method(c, k, γ, δ, β, α, A)
        plot_paths(paths, ss=k_ss, axes=axes)

The following calculation indicates that when  :math:`T` is very large,
the optimal capital stock is  close to
its steady state value most of the time.

.. code-block:: python3

    T_list = (250, 150, 50, 25)

    fix, axes = plt.subplots(1, 3, figsize=(13, 3))

    for T in T_list:
        c = np.zeros(T+1)
        k = np.zeros(T+2)
        c[0] = 0.3
        k[0] = k_init
        paths = bisection_method(c, k, γ, δ, β, α, A)
        plot_paths(paths, ss=k_ss, axes=axes)

Different colors in the above graphs are associated
different horizons :math:`T`.

Notice that as the horizon increases, the planner puts :math:`K_t`
closer to the steady state value :math:`\bar K` for longer.

This pattern reflects a **turnpike** property of the steady state.

A rule of thumb for the planner is

-  for whatever :math:`K_0` you start with, push :math:`K_t` toward
   the steady state and stay there for as long as you can.


The planner accomplishes this by adjusting the saving rate :math:`\frac{f(K_t) - C_t}{f(K_t)}`
over time.

Let's calculate the saving rate.

.. code-block:: python3

    @njit
    def S(K):
        '''Aggregate savings'''
        T = len(K) - 2
        S = np.zeros(T+1)
        for t in range(T+1):
            S[t] = K[t+1] - (1 - δ) * K[t]
        return S

    @njit
    def s(K):
        '''Savings rate'''
        T = len(K) - 2
        Y = f(A, K, α)
        Y = Y[0:T+1]
        s = S(K) / Y
        return s

    def plot_savings(paths, c_ss=None, k_ss=None, s_ss=None, axes=None):

        T = len(paths[0])
        k_star = paths[1]
        savings_path = s(k_star)
        new_paths = (paths[0], paths[1], savings_path)

        if axes is None:
            fix, axes = plt.subplots(1, 3, figsize=(13, 3))

        ylabels = ['$c_t$', '$k_t$', '$s_t$']
        titles = ['Consumption', 'Capital', 'Savings Rate']

        for path, y, title, ax in zip(new_paths, ylabels, titles, axes):
            ax.plot(path)
            ax.set(ylabel=y, title=title, xlabel='t')

        # Plot steady state value of consumption
        if c_ss is not None:
            axes[0].axhline(c_ss, c='k', ls='--', lw=1)

        # Plot steady state value of capital
        if k_ss is not None:
            axes[1].axhline(k_ss, c='k', ls='--', lw=1)

        # Plot steady state value of savings
        if s_ss is not None:
            axes[2].axhline(s_ss, c='k', ls='--', lw=1)

        axes[1].axvline(T, c='k', ls='--', lw=1)
        axes[1].scatter(T, k_star[-1], s=80)
        plt.tight_layout()

    T_list = (250, 150, 75, 50)

    fix, axes = plt.subplots(1, 3, figsize=(13, 3))

    for T in T_list:
        c = np.zeros(T+1)
        k = np.zeros(T+2)
        c[0] = 0.3
        k[0] = k_init
        paths = bisection_method(c, k, γ, δ, β, α, A)
        plot_savings(paths, k_ss=k_ss, axes=axes)


The Limiting Economy
--------------------------

We now consider an economy in which :math:`T = +\infty`.

The appropriate thing to do is to replace terminal condition
:eq:`constraint4` with

.. math::

  \lim_{T \rightarrow +\infty} \beta^T u'(C_T) K_{T+1} = 0

a condition that will be satisfied by a path that converges to an
optimal steady state.

We can approximate the optimal path from an arbitrary initial
:math:`K_0` and shooting towards the optimal steady state
:math:`K` at a large but finite :math:`T+1`.

In the following code, we do this for a large :math:`T`; we shoot
towards the **steady state** and plot consumption, capital and the
savings rate.

We know that in the steady state that the saving rate is constant
and that :math:`\bar s= \frac{f(\bar K)-\bar C}{f(\bar K)}`.

From :eq:`feasibility-constraint` the steady state saving rate equals

.. math::

  \bar s =\frac{ \delta \bar{K}}{f(\bar K)}

The steady state saving rate :math:`\bar S = \bar s f(\bar K)` is
the amount required to offset capital depreciation each period.

We first study optimal capital paths that start below the steady
state.

.. code-block:: python3

    T = 130

    # Steady states
    S_ss = δ * k_ss
    c_ss = f(A, k_ss, α) - S_ss
    s_ss = S_ss / f(A, k_ss, α)

    c = np.zeros(T+1)
    k = np.zeros(T+2)
    c[0] = 0.3
    k[0] = k_ss / 3         # Start below steady state
    paths = bisection_method(c, k, γ, δ, β, α, A, terminal=k_ss)
    plot_savings(paths, k_ss=k_ss, s_ss=s_ss, c_ss=c_ss)

Since :math:`K_0<\bar K`, :math:`f'(K_0)>\rho +\delta`.

The planner chooses a positive saving rate that is higher than  the steady state
saving rate. 

Note, :math:`f''(K)<0`, so as :math:`K` rises, :math:`f'(K)` declines.

The planner slowly lowers the savings rate until reaching a steady
state in which :math:`f'(K)=\rho +\delta`.

Exercise
---------

-  Plot the optimal consumption, capital, and savings paths when the
   initial capital level begins at 1.5 times the steady state level
   as we shoot towards the steady state at :math:`T=130`.

-  Why does the savings rate respond like it does?

Solution
----------

.. code-block:: python3

    T = 130

    c = np.zeros(T+1)
    k = np.zeros(T+2)
    c[0] = 0.3
    k[0] = k_ss * 1.5   # Start above steady state
    paths = bisection_method(c, k, γ, δ, β, α, A, terminal=k_ss)
    plot_savings(paths, k_ss=k_ss, s_ss=s_ss, c_ss=c_ss)

Concluding Remarks
========================

In this lecture  XXXX,  we study a decentralized version of an economy with exactly the same
technology and preference structure as deployed here.

In that lecture, we replace the  planner of this lecture with Adam Smith's **invisible hand**

In place of quantity choices made by the planner, there are market prices somewhat produced by 
the invisible hand. 

Market prices must adjust to reconcile distinct decisions that are made independently
by a representative household and a representative firm.

The relationship between a command economy like the one studied in this lecture and a market economy like that
studied in XXXX is a foundational topic in general equilibrium theory and welfare economics. 

