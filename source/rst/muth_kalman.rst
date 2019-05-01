.. _muth_kalman:

.. include:: /_static/includes/lecture_howto_py.raw

.. index::
    single: python

******************************
Reverse engineering a la Muth
******************************

.. contents:: :depth: 2

**Co-author: Chase Coleman**

In addition what's in Anaconda, this lecture uses the quantecon library 

.. code-block:: ipython
  :class: hide-output

  !pip install quantecon
  
We'll also need the following imports

.. code-block:: ipython

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.linalg as la
    
    from quantecon import Kalman
    from quantecon import LinearStateSpace
    from scipy.stats import norm
    
    %matplotlib inline
    np.set_printoptions(linewidth=120, precision=4, suppress=True)


This lecture uses the Kalman filter to reformulate John F. Muth’s first
paper about rational expectations

Muth used *classical* prediction methods to reverse engineer a
stochastic process that renders optimal Milton Friedman’s “adaptive
expectations” scheme

Friedman (1956) and Muth (1960)
=================================

Milton Friedman :cite:`Friedman1956` (1956) (consumption function book) posited that
consumer’s forecast their future disposable income with the adaptive
expectations scheme

.. math::
  :label: expectations
  
  y_{t+i,t}^* = K \sum_{j=0}^\infty (1 - K)^j y_{t-j}

where :math:`K \in (0,1)` and :math:`y_{t+i,t}^*` is a forecast of
future :math:`y` over horizon :math:`i`

Milton Friedman justified the **exponential smoothing** forecasting
scheme :eq:`expectations` informally, arguing that it seemed a plausible way to use
past income to forecast future income

In his first paper about rational expectations, John F. Muth :cite:`Muth1960`
reverse engineered a univariate stochastic process
:math:`\{y_t\}_{t=- \infty}^\infty` for which Milton Friedman’s adaptive
expectations scheme gives linear least forecasts of :math:`y_{t+j}` for
any horizon :math:`i`

Muth sought a setting and a sense in which Friedman’s forecasting scheme
is optimal

That is, Muth asked for what optimal forecasting **question** is Milton
Friedman’s adaptive expectation scheme the **answer**

Muth (1960) used classical prediction methods based on lag-operators and
:math:`z`-tranforms to find the answer to his question

Please see lectures XXXX and XXXX for an introduction to the classical
tools that Muth used

Rather than using those classical tools, in this lecture we apply the
Kalman filter to express the heart of Muth’s analysis concisely

A :math:`\{y_t\}` Process for Which Adaptive Expectations are Optimal
------------------------------------------------------------------------

Suppose that an observable :math:`y_t` is the sum of an unobserved
random walk :math:`x_t` and an i.i.d. shock :math:`\epsilon_{2,t}`:

.. math::
  :label: statespace
  
  \eqalign{ x_{t+1} & = x_t + \sigma_x \epsilon_{1,t+1} \cr
            y_t & = x_t + \sigma_y \epsilon_{2,t} }

where

.. math::  \left[\matrix{\epsilon_{1,t+1} \cr \epsilon_{2,t} } \right] \sim {\mathcal N} (0, I)

is an i.i.d. process

**Note:** A property of the statespace representation :eq:`statespace` is that in
general neither :math:`\epsilon_{1,t}` nor :math:`\epsilon_{2,t}` is in
the space spanned by square-summable lineary combinations of
:math:`y_t, y_{t-1}, \ldots`

In general
:math:`\begin{bmatrix} \epsilon_{1,t} \cr \epsilon_{2t} \end{bmatrix}`
has more information about future :math:`y_{t+j}`\ ’s than is contained
in :math:`y_t, y_{t-1}, \ldots`

We can use the asymptotic or stationary values of the Kalman gain and
the one-step ahead conditional state covarariance matrix to compute a
time-invariant *innovations representation*

.. math::
    :label: innovations

    \eqalign{ \hat x_{t+1} & = \hat x_t + K a_t  \cr
               y_t & = \hat x_t + a_t  }

where :math:`\hat x_t = E [x_t | y_{t-1}, y_{t-2}, \ldots ]` and
:math:`a_t = y_t - E[y_t |y_{t-1}, y_{t-2}, \ldots ]`

**Note:** A key property about an *innovations representation* is that
:math:`a_t` is in the space spanned by square summable linear
combinations of :math:`y_t, y_{t-1}, \ldots`

For more
ramifications of this property, see lecture XXXXX in the suite of DLE
lectures

Later we’ll stack these statespace systems :eq:`statespace` and :eq:`innovations` to display some
classic findings of Muth

But first let’s create an instance of the statespace system :eq:`statespace` then
apply the quantecon ``Kalman`` class

.. code-block:: python3

    # Make some parameter choices
    # sigx/sigy are state noise std err and measurement noise std err
    μ_0, σ_x, σ_y = 10, 1, 5
    
    # Create a LinearStateSpace object
    A, C, G, H = 1, σ_x, 1, σ_y
    ss = LinearStateSpace(A, C, G, H, mu_0=μ_0)
    
    # Set prior and initialize the Kalman type
    x_hat_0, Σ_0 = 10, 1
    kmuth = Kalman(ss, x_hat_0, Σ_0)
    
    # Computes stationary values which we need for the innovation representation
    S1, K1 = kmuth.stationary_values()
    
    # Form innovation representation state space
    Ak, Ck, Gk, Hk = A, K1, G, 1
    
    ssk = LinearStateSpace(Ak, Ck, Gk, Hk, mu_0=x_hat_0)

Some useful statespace math
-----------------------------

Now we want to map the time-invariant innovations representation :eq:`innovations` and
the original statespace system :eq:`statespace` into a convenient form for deducing
the impulse responses from the original shocks to the :math:`x_t` and
:math:`\hat x_t`

Putting both of these representations into a single state space system
is yet another application of the insight that “finding the state is an
art”

We’ll define a state vector and appropriate statespace matrices that
allow us to represent both systems in one fell swoop

Note that

.. math::  a_t = x_t + \sigma_y \epsilon_{2,t} - \hat x_t

so that

.. math::

   \eqalign{ \hat x_{t+1} & = \hat x_t + K (x_t + \sigma_y \epsilon_{2,t} - \hat x_t) \cr
          & = (1-K) \hat x_t + K x_t + K \sigma_y \epsilon_{2,t} } 

The stacked system

.. math::

    \left[ \matrix{ x_{t+1} \cr \hat x_{t+1} \cr \epsilon_{2,t+1} } \right] =
    \left[\matrix{ 1 & 0 & 0 \cr K & (1-K) & K \sigma_y \cr 0 & 0 & 0 } \right]
    \left[ \matrix{ x_{t} \cr \hat x_t \cr \epsilon_{2,t} } \right]+ 
    \left[ \matrix{ \sigma_x & 0 \cr 0 & 0 \cr 0 & 1} \right] 
    \left[ \matrix{ \epsilon_{1,t+1} \cr \epsilon_{2,t+1} } \right] 

.. math::

    \left[ \matrix{ y_t \cr a_t } \right] = \left[\matrix{ 1 & 0 & \sigma_y \cr
                                          1 & -1 & \sigma_y } \right]  \left[ \matrix{ x_{t} \cr \hat x_t \cr \epsilon_{2,t} } \right] 

is a statespace system that tells us how the shocks
:math:`\left[ \matrix{ \epsilon_{1,t+1} \cr \epsilon_{2,t+1} } \right]`
affect states :math:`\hat x_{t+1}, x_t`, the observable :math:`y_t`, and
the innovation :math:`a_t`

With this tool at our disposal, let’s form the composite system and
simulate it

.. code-block:: python3

    # Create grand state space for y_t, a_t as observed vars -- Use stacking trick above
    Af = np.array([[ 1,      0,        0], 
                   [K1, 1 - K1, K1 * σ_y], 
                   [ 0,      0,        0]])
    Cf = np.array([[σ_x,        0], 
                   [  0, K1 * σ_y], 
                   [  0,        1]])
    Gf = np.array([[1,  0, σ_y], 
                   [1, -1, σ_y]])
    
    μ_true, μ_prior = 10, 10
    μ_f = np.array([μ_true, μ_prior, 0]).reshape(3, 1)
    
    # Create the state space
    ssf = LinearStateSpace(Af, Cf, Gf, mu_0=μ_f)
    
    # Draw observations of y from state space model
    N = 50
    xf, yf = ssf.simulate(N)

    print(f"Kalman gain = {K1}")
    print(f"Conditional variance = {S1}")


Now that we have simulated our joint system, we have :math:`x_t`,
:math:`\hat{x_t}`, and :math:`y_t`

We can now investigate how these
variables are related by plotting some key objects

Estimates of unobservables
---------------------------

First, let’s plot the hidden state :math:`x_t` and the filtered version
:math:`\hat x_t` that is linear-least squares projection of :math:`x_t`
on the history :math:`y_{t-1}, y_{t-2}, \ldots`

.. code-block:: python3

    plt.plot(xf[0, :], label="$x_t$")
    plt.plot(xf[1, :], label="Filtered $x_t$")
    plt.legend()
    plt.xlabel("Time")
    plt.title(r"$x$ vs $\hat{x}$")
    plt.show()


Note how :math:`x_t` and :math:`\hat{x_t}` differ

For Friedman, :math:`\hat x_t` and not :math:`x_t` is the consumer’s
idea about her/his *permanent income*

Relation between unobservable and observable
---------------------------------------------

Now let’s plot :math:`x_t` and :math:`y_t`

Recall that :math:`y_t` is just :math:`x_t` plus white noise

.. code-block:: python3

    plt.plot(yf[0, :], label="y")
    plt.plot(xf[0, :], label="x")
    plt.legend()
    plt.title(r"$x$ and $y$")
    plt.xlabel("Time")
    plt.show()

We see above that :math:`y` seems to look like white noise around the
values of :math:`x`

Innovations
------------

Recall that we wrote down the innovation representation that depended on
:math:`a_t`. We now plot the innovations :math:`\{a_t\}`:

.. code-block:: python3

    plt.plot(yf[1, :], label="a")
    plt.legend()
    plt.title(r"Innovation $a_t$")
    plt.xlabel("Time")
    plt.show()


VAR/ARMA Representation
-----------------------

Now we shall extract from the **Kalman** instance kmuth coefficients of

-  a fundamental moving average representation that represents
   :math:`y_t` as a one-sided moving sum of current and past
   :math:`a_t`\ s

-  a univariate autogression representation that depicts the
   coefficients in a linear least squares projection of :math:`y_t` on
   the semi-infinite history :math:`y_{t-1}, y_{t-2}, \ldots`

Then we’ll plot each of them

.. code-block:: python3

    # Kalman Methods for MA and VAR
    coefs_ma = kmuth.stationary_coefficients(5, "ma")
    coefs_var = kmuth.stationary_coefficients(5, "var")

    # Coefficients come in a list of arrays; but we
    # want to plot them and so need to stack into array
    coefs_ma_array = np.vstack(coefs_ma)
    coefs_var_array = np.vstack(coefs_var)
    
    fig, ax = plt.subplots(2)
    ax[0].plot(coefs_ma_array, label="MA")
    ax[0].legend()
    ax[1].plot(coefs_var_array, label="VAR")
    ax[1].legend()
    
    plt.show()


The **moving average** coefficients in the top panel show tell-tale
signs of :math:`y_t` being a process whose first difference is a first
order autoregression

The **autoregressive coefficients** decline geometrically with decay
rate :math:`(1-K)`

These are exactly the target outcomes that Muth (1960) aimed to reverse
engineer

.. code-block:: python3

    print(f'decay parameter 1 - K1 = {1 - K1}')

