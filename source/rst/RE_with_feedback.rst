
.. _RE_with_feedback:

.. include:: /_static/includes/header.raw

.. highlight:: python3

**************************
A model of the price level
**************************

.. contents:: :depth: 2


In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon


.. code-block:: ipython

    import numpy as np
    import quantecon as qe
    import matplotlib.pyplot as plt
    %matplotlib inline
    from sympy import *
    init_printing()

Overview
==========

We study a rational expectations version of Philip Cagan’s model linking
the price level to the money supply.

Let

-  :math:`m_t^d` be the log of the demand for money

-  :math:`m_t` be the log of the supply of money

-  :math:`p_t` be the log of the price level

It follows that :math:`p_{t+1} - p_t` is the rate of inflation.

The logarithm of the demand for real money balances :math:`m_t^d - p_t`
is an inverse function of the expected rate of inflation
:math:`p_{t+1} - p_t` for :math:`t \geq 0`:

.. math::  m_t^d - p_t = - \beta (p_{t+1} - p_t ), \quad \beta >0

Equate the demand for log money :math:`m_t^d` to the supply of log money
:math:`m_t` in the above equation and rearrange to deduce that the
logarithm of the price level :math:`p_t` is related to the logarithm of
the money supply :math:`m_t` by

.. math::
  :label: equation_1

    p_t = (1 -\lambda) m_t + \lambda p_{t+1}

where :math:`\lambda \equiv \frac{\beta}{1+\beta} \in (0,1)`.

Solving the first order difference equation :eq:`equation_1` forward gives

.. math::
  :label: equation_2

    p_t = (1 - \lambda) \sum_{j=0}^\infty \lambda^j m_{t+j},

which is the unique **stable** solution of difference equation :eq:`equation_1` among
a class of more general solutions

.. math::  p_t = (1 - \lambda) \sum_{j=0}^\infty \lambda^j m_{t+j} + c \lambda^{-t}

that is indexed by the real number :math:`c \in {\bf R}`.

Because we want to focus on stable solutions, we set :math:`c=0`.

We begin by assuming that the log of the money supply is **exogenous**
in the sense that it is an autonomous process that does not feed back on
the log of the price level.

In particular, we assume that the log of the money supply is described
by the linear state space system

.. math::
  :label: equation_3

   \begin{aligned}
    m_t &  = G x_t \\ x_{t+1} & = A x_t
   \end{aligned}

where :math:`x_t` is an :math:`n \times 1` vector that does not include
:math:`p_t` or lags of :math:`p_t`, :math:`A` is an :math:`n \times n`
matrix with eigenvalues that are less than :math:`\lambda^{-1}` in
absolute values, and :math:`G` is a :math:`1 \times n` selector matrix.

Variables appearing in the vector :math:`x_t` contain information that
might help predict future values of the money supply.

We’ll take an example in which :math:`x_t` includes only :math:`m_t`,
possibly lagged values of :math:`m`, and a constant.

An example of such an :math:`\{m_t\}` process that fits info state space
system :eq:`equation_3` is one that satisfies the second order linear difference
equation

.. math::  m_{t+1} = \alpha + \rho_1 m_t + \rho_2 m_{t-1}

where the zeros of the characteristic polynomial
:math:`(1 - \rho_1 z - \rho_2 z^2)` are strictly greater than :math:`1`
in modulus

We seek a non-explosive solution of the difference equation :eq:`equation_1` that
obeys the system comprised of :eq:`equation_1`-:eq:`equation_3`.

By non-explosive, we mean that neither :math:`m_t` nor :math:`p_t`
diverges as :math:`t \rightarrow + \infty`.

The solution we are after is

.. math::
  :label: equation_4

    p_t = F x_t

where

.. math::
  :label: equation_5

   F = (1-\lambda) G (I - \lambda A)^{-1}

**Note:** As mentioned above, an *explosive solution* of difference
equation :eq:`equation_1` can be constructed by adding to the right hand of :eq:`equation_4` a
sequence :math:`c \lambda^{-t}` where :math:`c` is an arbitrary positive
constant.

Some Python code
================

We’ll construct examples that illustrate :eq:`equation_3`.

Our first example takes as the law of motion for the log money supply
the second order difference equation

.. math::
  :label: equation_6

    m_{t+1} = \alpha + \rho_1 m_t + \rho_2 m_{t-1}

that is parameterized by :math:`\rho_1, \rho_2, \alpha`

To capture this parameterization with system :eq:`equation_2` we set

.. math::

    x_t = \begin{bmatrix} 1 \cr m_t \cr m_{t-1} \end{bmatrix} , \quad
      A= \begin{bmatrix} 1 & 0 & 0 \cr
                         \alpha & \rho_1 & \rho_2 \cr
                          0 & 1 & 0 \end{bmatrix} , \quad
      G = \begin{bmatrix} 0 & 1 & 0 \end{bmatrix}

Here is Python code

.. code-block:: python3

    λ = .9

    α = 0
    ρ1 = .9
    ρ2 = .05

    A = np.array([[1,  0,  0],
                  [α, ρ1, ρ2],
                  [0,  1,  0]])
    G = np.array([[0, 1, 0]])

The matrix :math:`A` has one eigenvalue equal to unity.

It is associated with the :math:`A_{11}` component that captures a
constant component of the state :math:`x_t`.

We can verify that the two eigenvalues of :math:`A` not associated with
the constant in the state :math:`x_t` are strictly less than unity in
modulus.

.. code-block:: python3

    eigvals = np.linalg.eigvals(A)
    print(eigvals)

.. code-block:: python3

    (abs(eigvals) <= 1).all()

Now let’s compute :math:`F` in formulas :eq:`equation_4` and :eq:`equation_5`

.. code-block:: python3

    # compute the solution, i.e. forumula (3)
    F = (1 - λ) * G @ np.linalg.inv(np.eye(A.shape[0]) - λ * A)
    print("F= ",F)

Now let’s simulate paths of :math:`m_t` and :math:`p_t` starting from an
initial value :math:`x_0`.

.. code-block:: python3

    # set the initial state
    x0 = np.array([1, 1, 0])

    T = 100 # length of simulation

    m_seq = np.empty(T+1)
    p_seq = np.empty(T+1)

    m_seq[0] = G @ x0
    p_seq[0] = F @ x0

    # simulate for T periods
    x_old = x0
    for t in range(T):

        x = A @ x_old

        m_seq[t+1] = G @ x
        p_seq[t+1] = F @ x

        x_old = x

.. code-block:: python3

    plt.figure()
    plt.plot(range(T+1), m_seq, label='$m_t$')
    plt.plot(range(T+1), p_seq, label='$p_t$')
    plt.xlabel('t')
    plt.title(f'λ={λ}, α={α}, $ρ_1$={ρ1}, $ρ_2$={ρ2}')
    plt.legend()
    plt.show()

In the above graph, why is the log of the price level always less than
the log of the money supply?

The answer is because

-  according to equation :eq:`equation_2`, :math:`p_t` is a geometric weighted
   average of current and future values of :math:`m_t`, and

-  it happens that in this example future :math:`m`\ ’s are always less
   than the current :math:`m`

Alternative code
================

We could also have run the simulation using the quantecon
**LinearStateSpace** code

.. code-block:: python3

    # construct a LinearStateSpace instance

    # stack G and F
    G_ext = np.vstack([G, F])

    C = np.zeros((A.shape[0], 1))

    ss = qe.LinearStateSpace(A, C, G_ext, mu_0=x0)

.. code-block:: python3

    T = 100

    # simulate using LinearStateSpace
    x, y = ss.simulate(ts_length=T)

    # plot
    plt.figure()
    plt.plot(range(T+1), m_seq, label='$m_t$')
    plt.plot(range(T+1), p_seq, label='$p_t$')
    plt.xlabel('t')
    plt.title(f'λ={λ}, α={α}, $ρ_1$={ρ1}, $ρ_2$={ρ2}')
    plt.legend()
    plt.show()

Special case
^^^^^^^^^^^^^^^^^^^^^^

To simplify our presentation in ways that will let focus on an important
idea, in the above second-order difference equation :eq:`equation_6` that governs
:math:`m_t` we now set :math:`\alpha =0`,
:math:`\rho_1 = \rho \in (-1,1)`, and :math:`\rho_2 =0` so that the law
of motion for :math:`m_t` becomes

.. math::
  :label: equation_7

    m_{t+1} =\rho m_t

and the state :math:`x_t` becomes

.. math::  x_t = m_t .

So we can set :math:`G =1, A =\rho` making our formula :eq:`equation_5` for :math:`F`
become

.. math::  F = (1-\lambda) (1 -\lambda \rho)^{-1} .

and the log the log price level satisfies

.. math::  p_t = F m_t .

Please keep these formulas in mind as we investigate an alternative
route to and interpretation of the formula for :math:`F`.

Another perspective
===================

To shed light on the mechanics involved in imposing stability on a
solution of a potentially unstable system of linear difference equations
and to prepare the way for generalizations of our model in which the
money supply is allowed to feed back on the price level itself, we stack
equations :eq:`equation_1` and :eq:`equation_7` to form the system

.. math::
  :label: equation_8

    \begin{bmatrix} m_{t+1} \cr p_{t+1} \end{bmatrix} = \begin{bmatrix} \rho & 0 \\ - (1-\lambda)/\lambda & \lambda^{-1}  \end{bmatrix} \begin{bmatrix} m_t \\ p_t \end{bmatrix}

or

.. math::
  :label: equation_9

    y_{t+1} = H y_t, \quad t \geq 0

where

.. math::
  :label: equation_10

    H = \begin{bmatrix} \rho & 0 \\ - (1-\lambda)/\lambda & \lambda^{-1}  \end{bmatrix} .

Transition matrix :math:`H` has eigenvalues :math:`\rho \in (0,1)` and
:math:`\lambda^{-1} > 1`.

Because an eigenvalue of :math:`H` exceeds unity, if we iterate on
equation :eq:`equation_9` starting from an arbitrary initial vector
:math:`y_0 = \begin{bmatrix} m_0 \\ p_0 \end{bmatrix}`, we discover that
in general absolute values of both components of :math:`y_t` diverge
toward :math:`+\infty` as :math:`t \rightarrow + \infty`.

To substantiate this claim, we can use the eigenector matrix
decomposition of :math:`H` that is available to us because the
eigenvalues of :math:`H` are distinct

.. math::  H = Q \Lambda Q^{-1} .

Here :math:`\Lambda` is a diagonal matrix of eigenvalues of :math:`H`
and :math:`Q` is a matrix whose columns are eigenvectors of the
corresponding eigenvalues.

Note that

.. math::  H^t = Q \Lambda^t Q^{-1}

so that

.. math::  y_t = Q \Lambda^t Q^{-1} y_0

For almost all initial vectors :math:`y_0`, the presence of the
eigenvalue :math:`\lambda^{-1} > 1` causes both components of
:math:`y_t` to diverge in absolute value to :math:`+\infty`.

To explore this outcome in more detail, we use the following
transformation

.. math::  y^*_t = Q^{-1} y_t

that allows us to represent the dynamics in a way that isolates the
source of the propensity of paths to diverge:

.. math::  y^*_{t+1} = \Lambda^t y^*_t

Staring at this equation indicates that unless

.. math::
  :label: equation_11

    y^*_0 = \begin{bmatrix} y^*_{1,0} \cr 0 \end{bmatrix} ,

the path of :math:`y^*_t` and therefore the paths of both components of
:math:`y_t = Q y^*_t` will diverge in absolute value as
:math:`t \rightarrow +\infty`. (We say that the paths *explode*)

Equation :eq:`equation_11` also leads us to conclude that there is a unique setting
for the initial vector :math:`y_0` for which both components of
:math:`y_t` will not diverge.

The required setting of :math:`y_0` must evidently have the property
that

.. math::  Q y_0 =  y^*_0 = \begin{bmatrix} y^*_{1,0} \cr 0 \end{bmatrix} .

But note that since
:math:`y_0 = \begin{bmatrix} m_0 \cr p_0 \end{bmatrix}` and :math:`m_0`
is given to us an an initial condition, it has to be :math:`p_0` that
does all the adjusting to satisfy this equation.

Sometimes this situation is described by saying that while :math:`m_0`
is truly a **state** variable, :math:`p_0` is a **jump** variable that
is free to adjust at :math:`t=0` in order to satisfy the equation.

Thus, in a nutshell the unique value of the vector :math:`y_0` for which
the paths of :math:`y_t` do not diverge must have second component
:math:`p_0` that verifies equality :eq:`equation_11` by setting the second component
of :math:`y^*_0` equal to zero.

The component :math:`p_0` of the initial vector
:math:`y_0 = \begin{bmatrix} m_0 \cr p_0 \end{bmatrix}` must evidently
satisfy

.. math::  Q^{\{2\}} y_0 =0

where :math:`Q^{\{2\}}` denotes the second row of :math:`Q^{-1}`, a
restriction that is equivalent to

.. math::
  :label: equation_12

    Q^{21} m_0 + Q^{22} p_0 = 0

where :math:`Q^{ij}` denotes the :math:`(i,j)` component of
:math:`Q^{-1}`.

Solving this equation for :math:`p_0` we find

.. math::
  :label: equation_13

    p_0 = - (Q^{22})^{-1} Q^{21} m_0.

This is the unique **stabilizing value** of :math:`p_0` as a function of
:math:`m_0`.

Refining the formula
--------------------

We can get an even more convenient formula for :math:`p_0` that is cast
in terms of components of :math:`Q` instead of components of
:math:`Q^{-1}`.

To get this formula, first note that because :math:`(Q^{21}\ Q^{22})` is
the second row of the inverse of :math:`Q` and because
:math:`Q^{-1} Q = I`, it follows that

.. math:: \begin{bmatrix} Q^{21} & Q^{22} \end{bmatrix}  \begin{bmatrix} Q_{11}\cr Q_{21} \end{bmatrix} = 0

which implies that

.. math:: Q^{21} Q_{11} + Q^{22} Q_{21} = 0.

Therefore,

.. math:: -(Q^{22})^{-1} Q^{21} = Q_{21} Q^{-1}_{11}.

So we can write

.. math::
  :label: equation_14

    p_0 = Q_{21} Q_{11}^{-1} m_0 .

It can be verified that this formula replicates itself over time so that

.. math::
  :label: equation_15

    p_t = Q_{21} Q^{-1}_{11} m_t.

%

To implement formula :eq:`equation_15`, we want to compute :math:`Q_1` the
eigenvector of :math:`Q` associated with the stable eigenvalue
:math:`\rho` of :math:`Q`.

By hand it can be verified that the eigenvector associated with the
stable eigenvalue :math:`\rho` is proportional to

.. math::  Q_1  = \begin{bmatrix} 1-\lambda  \rho \\ 1 - \lambda   \end{bmatrix}.

Notice that if we set :math:`A=\rho` and :math:`G=1` in our earlier
formula for :math:`p_t` we get

.. math::  Q = G (I - \lambda A)^{-1} m_t =  (1-\lambda) (1 - \lambda \rho)^{-1} m_t

a formula that is equivalent with

.. math::  p_t = Q_{21} Q_{11}^{-1}  m_t ,

where

.. math::  Q_1 = \begin{bmatrix} Q_{11} \\ Q_{21}  \end{bmatrix}.

Some remarks about feedback
---------------------------

We have expressed :eq:`equation_8` in what superficially appears to be a form in
which :math:`y_{t+1}` feeds back on :math:`y_t`. even though what we
actually want to represent is that the component :math:`p_t` feeds
**forward** on :math:`p_{t+1}`, and through it, on future
:math:`m_{t+j}`, :math:`j = 0, 1, 2, \ldots`.

A tell-tale sign that we should look beyond its superficial “feedback”
form is that :math:`\lambda^{-1} > 1` so that the matrix :math:`H` in
:eq:`equation_8` is **unstable**

-  it has one eigenvalue :math:`\rho` that is less than one in modulus
   that does not imperil stability, but :math:`\ldots`

-  it has a second eigenvalue :math:`\lambda^{-1}` that exceeds one in
   modulus and that makes :math:`H` an unstable matrix

We’ll keep these observations in mind as we turn now to a case in which
the log money supply actually does feed back on the log of the price
level.

Log money supply feeds back on log price level
==============================================

The same pattern of eigenvalues splitting around unity, with one being
below unity and another greater than unity, will sometimes continue to
prevail when we activate feedback from the log price level to the log
money supply.

Let the feedback rule be

.. math::
  :label: equation_16

    m_{t+1} =  \rho m_t + \delta p_t

where :math:`\rho \in (0,1)` as before and where we shall now allow
:math:`\delta \neq 0`.

However, we shall see that we have to be careful about allowing
:math:`\delta` to be too large in order for things to fit together as we
wish to deliver a stable system for some initial condition :math:`p_0`.
.

The forward-looking equation :eq:`equation_1` continues to describe equality between
the demand and supply of money.

We assume that equations :eq:`equation_1` and :eq:`equation_16` govern
:math:`y_t \equiv \begin{bmatrix} m_t \cr p_t \end{bmatrix}` for
:math:`t \geq 0`

The transition matrix :math:`H` in the law of motion

.. math::  y_{t+1} = H y_t

now becomes

.. math::  H = \begin{bmatrix} \rho & \delta \\ - (1-\lambda)/\lambda & \lambda^{-1}  \end{bmatrix}

We take :math:`m_0` as a given intial condition and as before seek an
initial value :math:`p_0` that stabilizes the system in the sense that
:math:`y_t` converges as :math:`t \rightarrow + \infty`.

Our approach is identical with that followed above and is based on an
eigenvalue decomposition in which, cross our fingers, one eigenvalue
exceeds unity and the other is less than unity in absolute value.

When :math:`\delta \neq 0` as we now assume, the eigenvalues of
:math:`H` are no longer :math:`\rho \in (0,1)` and
:math:`\lambda^{-1} > 1`

We’ll just calculate them and apply the same algorithm that we used
above.

That algorithm remains valid so long as the eigenvalues split around
unity as before.

Again we assume that :math:`m_0` is as an initial condition, but that
:math:`p_0` is not given but to be solved for.

Let’s write some code that will let us explore how outcomes depend on
:math:`\delta`.

.. code-block:: python3

    def construct_H(ρ, λ, δ):
        "contruct matrix H given parameters."

        H = np.empty((2, 2))
        H[0, :] = ρ,δ
        H[1, :] = - (1 - λ) / λ, 1 / λ

        return H

    def H_eigvals(ρ=.9, λ=.5, δ=0):
        "compute the eigenvalues of matrix H given parameters."

        # construct H matrix
        H = construct_H(ρ, λ, δ)

        # compute eigenvalues
        eigvals = np.linalg.eigvals(H)

        return eigvals

.. code-block:: python3

    H_eigvals()

Notice that a negative δ will not imperil the stability of the matrix
:math:`H`, even if it has a big absolute value.

.. code-block:: python3

    # small negative δ
    H_eigvals(δ=-0.05)

.. code-block:: python3

    # large negative δ
    H_eigvals(δ=-1.5)

A sufficiently small positive δ also causes no problem.

.. code-block:: python3

    # sufficiently small positive δ
    H_eigvals(δ=0.05)

But a large enough positive δ makes both eigenvalues of :math:`H`
strictly greater than unity in modulus.

For example,

.. code-block:: python3

    H_eigvals(δ=0.2)

We want to study systems in which one eigenvalue exceeds unity in
modulus while the other is less than unity in modulus, so we avoid
values of :math:`\delta` that are too large

.. code-block:: python3

    def magic_p0(m0, ρ=.9, λ=.5, δ=0):
        """
        Use the magic formula (8) to compute the level of p0
        that makes the system stable.
        """

        H = construct_H(ρ, λ, δ)
        eigvals, Q = np.linalg.eig(H)

        # find the index of the smaller eigenvalue
        ind = 0 if eigvals[0] < eigvals[1] else 1

        # verify that the eigenvalue is less than unity
        if eigvals[ind] > 1:

            print("both eigenvalues exceed unity in modulus")

            return None

        p0 = Q[1, ind] / Q[0, ind] * m0

        return p0

First, we plot how the solution :math:`p_0` changes as :math:`m_0`
changes for different settings of :math:`\delta`.

.. code-block:: python3

    m_range = np.arange(0.1, 2., 0.1)

    for δ in [-0.05, 0, 0.05]:
        plt.plot(m_range, [magic_p0(m0, δ=δ) for m0 in m_range], label=f"δ={δ}")
    plt.legend()

    plt.xlabel("$m_0$")
    plt.ylabel("$p_0$")
    plt.show()

From another point of view, we can fix the initial value :math:`m_0` and
see how :math:`p_0` changes as :math:`\delta` changes.

.. code-block:: python3

    m0 = 1

    δ_range = np.linspace(-0.05, 0.05, 100)
    plt.plot(δ_range, [magic_p0(m0, δ=δ) for δ in δ_range])
    plt.xlabel('$\delta$')
    plt.ylabel('$p_0$')
    plt.title(f'$m_0$={m0}')
    plt.show()

Notice that when :math:`\delta` is large enough, both eigenvalues exceed
unity in modulus, causing a stabilizing value of :math:`p_0` not to
exist.

.. code-block:: python3

    magic_p0(1, δ=0.2)

More requests for Zejin, Feb 19, 2020
=====================================

Please do the following calculations.

1. For the system with :math:`\delta\neq 0` so that there is feedback,
   please compute the stabilizing solution for :math:`p_t` in the form
   :math:`p_t = F^* m_t` where :math:`F^* = Q_{21}Q_{11}^{-1}` as above.

2. Please recall the system :eq:`equation_3`, :eq:`equation_4`, and :eq:`equation_5` above. Please define
   :math:`x_t = \begin{bmatrix} m_t \cr P_t \end{bmatrix}` and notice
   that it is Big :math:`P_t` and not little :math:`p_t` here. This is
   not a typo. Please form :math:`A` and :math:`G` as
   :math:`A = \begin{bmatrix}\rho & \delta \cr F^* \rho & F^*\delta \end{bmatrix}`
   and :math:`G = \begin{bmatrix} 1 & 0 \end{bmatrix}` and then compute
   :math:`\check F = \begin{bmatrix} \check F_1 & \check F_2 \end{bmatrix} \equiv F`
   from equation :eq:`equation_5` above.

3. Please compute :math:`\check F_1 + \check F_2 F^*` and compare it
   with :math:`F^*`.

.. code-block:: python3

    # set parameters
    ρ = .9
    λ = .5
    δ = .05

.. code-block:: python3

    # solve for F_star
    H = construct_H(ρ, λ, δ)
    eigvals, Q = np.linalg.eig(H)

    ind = 0 if eigvals[0] < eigvals[1] else 1
    F_star = Q[1, ind] / Q[0, ind]
    F_star

.. code-block:: python3

    # solve for F_check
    A = np.empty((2, 2))
    A[0, :] = ρ, δ
    A[1, :] = F_star * A[0, :]

    G = np.array([1, 0])

    F_check= (1 - λ) * G @ np.linalg.inv(np.eye(2) - λ * A)
    F_check

Compare :math:`F^*` with :math:`\check F_1 + \check F_2 F^*`

.. code-block:: python3

    F_check[0] + F_check[1] * F_star, F_star


Feb. 2: Adding Sympy code
=========================

:math:`H` with nonzero :math:`\delta`.

.. code-block:: python3

    λ, δ, ρ = symbols('λ, δ, ρ')

.. code-block:: python3

    H1 = Matrix([[ρ,δ], [- (1 - λ) / λ, λ ** -1]])

.. code-block:: python3

    H1

.. code-block:: python3

    H1.eigenvals()

.. code-block:: python3

    H1.eigenvects()

:math:`H` with :math:`\delta` being zero.

.. code-block:: python3

    H2 = Matrix([[ρ,0], [- (1 - λ) / λ, λ ** -1]])

.. code-block:: python3

    H2

.. code-block:: python3

    H2.eigenvals()

.. code-block:: python3

    H2.eigenvects()

Requests for Zejin, Jan 24
==========================

May I please ask that you do the following with this great code?

1. In sympy, please compute the matrix :math:`Q` whose first column is
   the eigenvector associated with :math:`\rho`. and whose second column
   is the eigenvector associated with :math:`\lambda^{-1}`.

2. Please use sympy to compute the inverse :math:`Q^{-1}` of :math:`Q`
   (both in symbols).

3. Please use sympy to compute :math:`Q_{21} Q_{11}^{-1}` (in symbols).

4. Where :math:`Q^{ij}` denotes the :math:`(i,j)` component of
   :math:`Q^{-1}`, please use sympy to compute
   :math:`- (Q^{22})^{-1} Q^{21}` (again in symbols)

If you can do these things in sympy, some magic will happen.

After this has been done, we can bring this code into the “main” Cagan
notebook. But let’s wait please for that

This notebook is really shaping up and will be great fun to teach in
class.

.. code-block:: python3

    # construct Q
    vec = []
    for i, (eigval, _, eigvec) in enumerate(H2.eigenvects()):

        vec.append(eigvec[0])

        if eigval == ρ:
            ind = i

    Q = vec[ind].col_insert(1, vec[1-ind])

.. code-block:: python3

    Q

:math:`Q^{-1}`

.. code-block:: python3

    Q_inv = Q ** (-1)
    Q_inv

:math:`Q_{21}Q_{11}^{-1}`

.. code-block:: python3

    Q[1, 0] / Q[0, 0]

:math:`−(Q^{22})^{−1}Q^{21}`

.. code-block:: python3

    - Q_inv[1, 0] / Q_inv[1, 1]
