.. _tax_smoothing_1:

.. include:: /_static/includes/lecture_howto_py.raw

.. index::
    single: python


****************************
How to Pay for a War: Part 1
****************************

.. contents:: :depth: 2

**Co-author:** `Sebastian Graves <https://github.com/sebgraves>`__


Reader's Guide
===================

This lecture ------------


  1. :doc:`How to Pay for a War: Part 2 <tax_smoothing_2>`

  2. :doc:`How to Pay for a War: Part 3 <tax_smoothing_3>`

  3. :doc:` < >`

  4. :doc:` < >`


An Application of Markov Jump Linear Quadratic Dynamic Programming
==================================================================


This lecture constructs generalizations of Barro’s classic 1979 :cite:`Barro1979` model
of tax smoothing

Our generalizations are adaptations of extensions of
his 1979 model suggested by Barro (1999 :cite:`barro1999determinants`, 2003 :cite:`barro2003religion`)

Barro’s original 1979 :cite:`Barro1979` model is about a government that borrows and lends
in order to help it minimize an intertemporal measure of distortions
caused by taxes

Technical tractability induced Barro to assume that

-  the government trades only one-period risk-free debt, and

-  the one-period risk-free interest rate is constant

By using a secret weapon – *Markov jump linear quadratic dynamic
programming* – we can allow interest rates to move over time in
empirically interesting ways

Also, by expanding the dimension of the
state, we can add a maturity composition decision to the government’s
problem

It is by doing these two things that we extend Barro’s 1979 :cite:`Barro1979`
model along lines he suggested in Barro (1999 :cite:`barro1999determinants`, 2003 :cite:`barro2003religion`)

Barro (1979) :cite:`Barro1979` assumed

-  that a government faces an **exogenous sequence** of expenditures
   that it must finance by a tax collection sequence whose expected
   present value equals the initial debt it owes plus the expected
   present value of those expenditures

-  that the government wants to minimize the following measure of tax
   distortions: :math:`E_0 \sum_{t=0}^{\infty} \beta^t T_t^2`, where :math:`T_t` are total tax collections and :math:`E_0`
   is a mathematical expectation conditioned on time :math:`0`
   information

-  that the government trades only one asset, a risk-free one-period
   bond

-  that the gross interest rate on the one-period bond is constant and
   equal to :math:`\beta^{-1}`, the reciprocal of the factor
   :math:`\beta` at which the government discounts future tax disortions

Barro’s model can be mapped into a discounted linear quadratic dynamic
programming problem

Our generalizations of Barro’s (1979) :cite:`Barro1979` model, partly inspired by Barro
(1999) :cite:`barro1999determinants` and Barro (2003) :cite:`barro2003religion`, assume

-  that the government borrows or saves in the form of risk-free bonds
   of maturities :math:`1, 2, \ldots , H`

-  that interest rates on those bonds are time-varying and in particular
   governed by a jointly stationary stochastic process

Our generalizations are designed to fit within a generalization of an
ordinary linear quadratic dynamic programming problem in which matrices
defining the quadratic objective function and the state transition
function are **time-varying** and **stochastic**

This generalization, known as a **Markov jump linear quadratic dynamic
program** combines

-  the computational simplicity of **linear quadratic dynamic
   programming**, and

-  the ability of **finite state Markov chains** to represent
   interesting patterns of random variation

We want the stochastic time variation in the matrices defining the
dynamic programming problem to represent variation over time in

-  interest rates

-  default rates

-  roll over risks

The idea underlying **Markov jump linear quadratic dynamic programming**
is to replace the constant matrices defining a **linear quadratic
dynamic programming problem** with matrices that are fixed functions of
an :math:`N` state Markov chain

For infinite horizon problems, this leads to :math:`N` interrelated
matrix Riccati equations that pin down :math:`N` value functions and
:math:`N` linear decision rules, applying to the :math:`N` Markov
states

Public Finance Questions
^^^^^^^^^^^^^^^^^^^^^^^^

Barro’s 1979 :cite:`Barro1979` model is designed to answer questions such as

-  Should a government finance an exogenous surge in government
   expenditures by raising taxes or borrowing?

-  How does the answer to that first question depend on the exogenous
   stochastic process for government expenditures, for example, on
   whether the surge in government expenditures can be expected to be
   temporary or permanent?

Barro’s 1999 :cite:`barro1999determinants` and 2003 :cite:`barro2003religion`
models are designed to answer more fine-grained
questions such as

-  What determines whether a government wants to issue short-term or
   long-term debt?

-  How do roll-over risks affect that decision?

-  How does the government’s long-short *portfolio management* decision
   depend on features of the exogenous stochastic process for government
   expenditures?

Thus, both the simple and the more fine-grained versions of Barro’s
models are ways of precisely formulating the classic issue of *How to
pay for a war*

Organization
^^^^^^^^^^^^

This lecture describes:

-  Markov jump linear quadratic (LQ) dynamic programming

-  An application of Markov jump LQ dynamic programming to a model in
   which a government faces exogenous time-varying interest rates for
   issuing one-period risk-free debt

A :doc:`sequel to this
lecture <tax_smoothing_2>`
describes applies Markov LQ control to settings in which a government
issues risk-free debt of different maturities

Markov Jump Linear Quadratic Control
====================================

**Markov jump linear quadratic dynamic programming** combines advantages
of

-  the computational simplicity of **linear quadratic dynamic
   programming**, and

-  the ability of **finite state Markov chains** to represent
   interesting patterns of random variation

The idea underlying **Markov jump linear quadratic dynamic programming**
is to replace the constant matrices defining a **linear quadratic
dynamic programming problem** with matrices that are fixed functions of
an :math:`N` state Markov chain

For infinite horizon problems, this leads to :math:`N` interrelated
matrix Riccati equations that determine :math:`N` optimal value
functions and :math:`N` linear decision rules

These value functions and
decision rules apply in the :math:`N` Markov states: i.e. when the
Markov state is in state :math:`j`, the value function and the decision rule
for state :math:`j` prevails

The Ordinary Discounted Linear Quadratic Dynamic Programming Problem
--------------------------------------------------------------------

It is handy to have the following reminder in mind

A **linear quadratic dynamic programming problem** consists of a scalar
discount factor :math:`\beta \in (0,1)`, an :math:`n\times 1` state
vector :math:`x_t`, an initial condition for :math:`x_0`, a
:math:`k \times 1` control vector :math:`u_t`, a :math:`p \times 1`
random shock vector :math:`w_{t+1}` and the following two triples of
matrices:

-  A triple of matrices :math:`(R, Q, W)` defining a loss function

.. math::  r(x_t, u_t) = x_t' R x_t + u_t' Q u_t + 2 u_t' W x_t

-  a triple of matrices :math:`(A, B, C)` defining a state-transition
   law

.. math::  x_{t+1} = A x_t + B u_t + C w_{t+1}

The problem is

.. math::


   -x_0' P x_0 - \rho = \min_{\{u_t\}_{t=0}^\infty} E \sum_{t=0}^{\infty} \beta^t r(x_t, u_t)

subject to the transition law for the state

The optimal decision rule for this problem has the form

.. math::  u_t = - F x_t

and the optimal value function is of the form

.. math::  -\left( x_t' P x_t  + \rho \right)

where :math:`P` solves the algebraic matrix Riccati equation

.. math::


   P = R+ \beta A' P A_i
             -(\beta B'  P A + W)' (Q + \beta B P B )^{-1} (\beta B P A + W)

and the constant :math:`\rho` satisfies

.. math::

   \rho = \beta
     \left( \rho + {\rm trace}(P C C') \right)

and the matrix :math:`F` in the decision rule for :math:`u_t` satisfies

.. math::


   F = (Q + \beta  B' P B)^{-1} (\beta (B' P A )+ W)

Markov Jump Coefficients
^^^^^^^^^^^^^^^^^^^^^^^^

The idea is to make the matrices :math:`A, B, C, R, Q, W` fixed
functions of a finite state :math:`s` that is governed by an :math:`N`
state Markov chain

This makes decision rules depend on the Markov
state, and so fluctuate through time restricted ways

In particular, we use the following extension of a discrete time linear
quadratic dynamic programming problem

We let :math:`s(t) \equiv s_t \in [1, 2, \ldots, N]` be a time :math:`t` realization of an
:math:`N` state Markov chain with transition matrix :math:`\Pi` having
typical element :math:`\Pi_{ij}`

Here :math:`i` denotes today and
:math:`j` denotes tomorrow and

.. math::  \Pi_{ij} = {\rm Prob}(s_{t+1} = j |s_t = i)

We’ll switch between labeling today’s state as :math:`s(t)` and
:math:`i` and between labeling tomorrow’s state as :math:`s(t+1)` or
:math:`j`

The decision maker solves the minimization problem:

.. math::

  \min_{\{u_t\}_{t=0}^\infty} E \sum_{t=0}^{\infty} \beta^t r(x_t, s(t), u_t)


with

.. math::

  r(x_t, s(t), u_t) = -( x_t' R(s_t) x_t + u_t' Q(s_t) u_t + 2 u_t' W(s_t) x_t)


subject to linear laws of motion with matrices :math:`(A,B,C)` each
possibly dependent on the Markov-state-\ :math:`s_t`:

.. math::


    x_{t+1} = A(s_t) x_t + B(s_t) u_t + C(s_t) w_{t+1}

where :math:`\{w_{t+1}\}` is an i.i.d. stochatic process with
:math:`w_{t+1} \sim {\cal N}(0,I)`

The optimal decision rule for this problem has the form

.. math::  u_t = - F(s_t) x_t

and the optimal value functions are of the form

.. math::  -\left( x_t' P(s_t) x_t  + \rho(s_t) \right)

or equivalently

.. math::  -x_t' P_i x_t - \rho_i

The optimal value functions :math:`- x' P_i x - \rho_i` for
:math:`i = 1, \ldots, n` satisfy the :math:`N`
interrelated Bellman equations

.. math::

    -x' P_i x - \rho_i  = \max_u - \biggl[ x'R_i x + u' Q_i u + 2 u' W_i x \\
                    \beta \sum_j \Pi_{ij}E ((A_i x + B_i u + C_i w)' P_j
                    (A_i x + B_i u + C_i w) x + \rho_j) \biggr]


The matrices :math:`P(s(t)) = P_i` and the scalars
:math:`\rho(s_t) = \rho_i, i = 1, \ldots`, n satisfy the following stacked system of
**algebraic matrix Riccati** equations:

.. math::


   P_i = R_i + \beta \sum_j A_i' P_j A_i
    \Pi_{ij}
             -\sum_j \Pi_{ij}[ (\beta B_i'  P_j A_i + W_i)' (Q + \beta B_i' P_j B_i)^{-1}
             (\beta B_i' P_j A_i + W_i)]

.. math::

   \rho_i = \beta
    \sum_j \Pi_{ij} ( \rho_j + {\rm trace}(P_j C_i C_i') )

and the :math:`F_i` in the optimal decision rules are

.. math::


   F_i = (Q_i + \beta \sum_j \Pi_{ij} B_i' P_j B_i)^{-1}
   (\beta \sum_j \Pi_{ij}(B_i' P_j A_i )+ W_i)


Barro (1979) Model
==================

We begin by solving a version of the Barro (1979) :cite:`Barro1979` model by mapping it
into the original LQ framework

As mentioned :doc:`in this lecture <perm_income_cons>`, the
Barro model is mathematically isomorphic with the LQ permanent income
model

Let :math:`T_t` denote tax collections, :math:`\beta` a discount factor,
:math:`b_{t,t+1}` time :math:`t+1` goods that the government promises to
pay at :math:`t`, :math:`G_t` government purchases, :math:`p_{t,t+1}`
the number of time :math:`t` goods received per time :math:`t+1` goods
promised

Evidently, :math:`p_{t, t+1}` is inversely related to
appropriate corresponding gross interest rates on government debt

In the spirit of Barro (1979) :cite:`Barro1979`, the stochastic process of government
expenditures is exogenous

The government’s problem is to choose a plan
for taxation and borrowing :math:`\{b_{t+1}, T_t\}_{t=0}^\infty` to
minimize


.. math::  E_0 \sum_{t=0}^\infty \beta^t T_t^2

subject to the constraints

.. math::  T_t + p_{t,t+1} b_{t,t+1} = G_t + b_{t-1,t}

.. math:: G_t = U_{g,t} z_t

.. math::  z_{t+1} = A_{22,t} z_t + C_{2,t} w_{t+1}

where :math:`w_{t+1} \sim {\cal N}(0,I)`

The variables
:math:`T_t, b_{t, t+1}` are *control* variables chosen at :math:`t`,
while :math:`b_{t-1,t}` is an endogenous state variable inherited from
the past at time :math:`t` and :math:`p_{t,t+1}` is an exogenous state
variable at time :math:`t`

To begin with, we will assume that
:math:`p_{t,t+1}` is constant (and equal to :math:`\beta`), but we will
also extend the model to allow this variable to evolve over time

To map into the LQ framework, we will use
:math:`x_t = \begin{bmatrix} b_{t-1,t} \\ z_t \end{bmatrix}` as the
state vector, and :math:`u_t = b_{t,t+1}` as the control variable

Therefore, the :math:`(A, B, C)` matrices are defined by the state-transition law:

.. math::  x_{t+1} = \begin{bmatrix} 0 & 0 \\ 0 & A_{22} \end{bmatrix} x_t + \begin{bmatrix} 1 \\ 0 \end{bmatrix} u_t + \begin{bmatrix} 0 \\ C_2 \end{bmatrix} w_{t+1}

To find the appropriate :math:`(R, Q, W)` matrices, we note that :math:`G_t` and
:math:`b_{t-1,t}` can be written as appropriately defined functions of
the current state:

.. math::  G_t = S_G x_t \hspace{2mm}, \hspace{2mm} b_{t-1,t} = S_1 x_t

If we define :math:`M_t = - p_{t,t+1}`, and let $S = S_G + S_1 $, then
we can write taxation as a function of the states and control using the
government’s budget constraint:

.. math::  T_t = S x_t + M_t u_t

It follows that the :math:`(R, Q, W)` matrices are implicitly defined by:

.. math::  T_t^2 = x_t'S'Sx_t + u_t'M_t'M_tu_t + 2 u_t'M_t'S x_t

If we assume that :math:`p_{t,t+1} = \beta`, then :math:`M_t \equiv M = -\beta`

In this case, none of
the LQ matrices are time varying, and we can use the original LQ
framework

We will implement this constant interest-rate version first, assuming
that :math:`G_t` follows an AR(1) process:

.. math::  G_{t+1} = \bar G + \rho G_t + \sigma w_{t+1}

To do this, we set
:math:`z_t = \begin{bmatrix} 1 \\ G_t \end{bmatrix}`, and consequently:

.. math::

  A_{22} = \begin{bmatrix} 1 & 0 \\ \bar G & \rho \end{bmatrix} \hspace{2mm} ,
  \hspace{2mm} C_2 = \begin{bmatrix} 0 \\ \sigma \end{bmatrix}

.. code-block:: ipython

    import quantecon as qe
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline

.. code-block:: python3

    # Model parameters
    β, Gbar, ρ, σ = 0.95, 5, 0.8, 1

    # Basic model matrices
    A22 = np.array([[1,    0],
                    [Gbar, ρ],])

    C2 = np.array([[0],
                   [σ]])

    Ug = np.array([[0, 1]])

    # LQ framework matrices
    A_t = np.zeros((1, 3))
    A_b = np.hstack((np.zeros((2, 1)), A22))
    A = np.vstack((A_t, A_b))

    B = np.zeros((3, 1))
    B[0, 0] = 1

    C = np.vstack((np.zeros((1, 1)), C2))

    Sg = np.hstack((np.zeros((1, 1)), Ug))
    S1 = np.zeros((1, 3))
    S1[0, 0] = 1
    S = S1 + Sg

    M = np.array([[-β]])

    R = S.T @ S
    Q = M.T @ M
    W = M.T @ S

    # Small penalty on the debt required to implement the no-Ponzi scheme
    R[0, 0] = R[0, 0] + 1e-9

We can now create an instance of ``LQ``:

.. code-block:: python3

    LQBarro = qe.LQ(Q, R, A, B, C=C, N=W, beta=β)
    P, F, d = LQBarro.stationary_values()
    x0 = np.array([[100, 1, 25]])

We can see the isomorphism by noting that consumption is a martingale in
the permanent income model and that taxation is a martingale in Barro’s
model

We can check this using the :math:`F` matrix of the LQ model

As :math:`u_t = -F x_t`, we have:

.. math::  T_t = S x_t + M u_t = (S - MF) x_t

and

.. math::  T_{t+1} = (S-MF)x_{t+1} = (S-MF)(Ax_t + B u_t + C w_{t+1}) = (S-MF)((A-BF)x_t + C w_{t+1})

Therefore, the conditional expectation of :math:`T_{t+1}` at time
:math:`t` is:

.. math::  E_t T_{t+1} = (S-MF)(A-BF)x_t

Consequently, taxation is a martingale (:math:`E_t T_{t+1} = T_t`) if:

.. math:: (S-MF)(A-BF) = (S-MF)

which holds in this case:

.. code-block:: python3

    S - M @ F, (S - M @ F) @ (A - B @ F)

This explains the gradual fanning out of taxation if we simulate the
Barro model a large number of times:

.. code-block:: python3

    T = 500
    for i in range(250):
        x, u, w = LQBarro.compute_sequence(x0, ts_length=T)
        plt.plot(list(range(T+1)), ((S - M @ F) @ x)[0, :])
    plt.xlabel('Time')
    plt.ylabel('Taxation')
    plt.show()

We can see a similar, but a smoother pattern, if we plot government debt
over time

Debt is smoother due to the persistence of the government
spending process

.. code-block:: python3

    T = 500
    for i in range(250):
        x, u, w = LQBarro.compute_sequence(x0, ts_length=T)
        plt.plot(list(range(T+1)), x[0, :])
    plt.xlabel('Time')
    plt.ylabel('Taxation')
    plt.show()

Python Class to Solve Markov Jump Linear Quadratic Control Problems
===================================================================

To implement the extension to the Barro model in which :math:`p_{t,t+1}`
varies over time, we must allow the M matrix to be time-varying

From
the mapping of the Barro model into the LQ framework, this means that
our :math:`Q` and :math:`W` matrices will now also vary over time

We can solve such a
model using the ``LQ_Markov`` class, which solves Markov jump linear
quandratic control problems as described above

The code for the class can be viewed
`here <https://github.com/QuantEcon/QuantEcon.notebooks/blob/master/dependencies/lq_markov.py>`__

The class takes a variable number of arguments, to allow for there to be
an arbitrary :math:`N` Markov states

To accomodate this, the
matrices for each Markov state must be held in a ``namedtuple``

The value and policy functions are then found by iterating on the system of
algebraic matrix Riccati equations

The solutions for :math:`P,F,\rho` are stored in Python “dictionaries”

The class also contains a “method”, for simulating the model

This is an
extension of a similar method in the ``LQ`` class, adapted to take into
account the fact that the model’s matrices depend on the Markov  state

The code below runs
`this file <https://github.com/QuantEcon/QuantEcon.notebooks/blob/master/dependencies/lq_markov.py>`_
containing the class and function we need using QuantEcon.py's
``fetch_nb_dependencies`` function

.. code-block:: ipython

    from quantecon.util.notebooks import fetch_nb_dependencies
    fetch_nb_dependencies(['lq_markov.py'],
                          repo='https://github.com/QuantEcon/QuantEcon.notebooks',
                          folder='dependencies')
    %run lq_markov.py


Barro Model with a Time-varying Interest Rate
=============================================

We can use the above class to implement a version of the Barro model
with a time-varying interest rate. The simplest way to extend the model
is to allow the interest rate to take two possible values. We set:

.. math::  p^1_{t,t+1} = \beta + 0.02 = 0.97

.. math::  p^2_{t,t+1} = \beta - 0.017 = 0.933

Thus, the first Markov state  has a low-interest rate, and the
second Markov state has a high-interest rate

We also need to specify a transition matrix for the Markov state

we use:

.. math::  \Pi = \begin{bmatrix} 0.8 & 0.2 \\ 0.2 & 0.8 \end{bmatrix}

(so each Markov state is persisent, and there is an equal chance
of moving from one state to the other)

The choice of parameters means that the unconditional expectation of
:math:`p_{t,t+1}` is 0.9515, higher than :math:`\beta (=0.95)`

If we
were to set :math:`p_{t,t+1} = 0.9515` in the version of the model with
a constant interest rate, government debt would explode

.. code-block:: python3

    # Create namedtuple to keep the R, Q, A, B, C, W matrices for each Markov state
    world = namedtuple('world', ['A', 'B', 'C', 'R', 'Q', 'W'])

    Π = np.array([[0.8, 0.2],
                  [0.2, 0.8]])

    M1 = np.array([[-β - 0.02]])
    M2 = np.array([[-β + 0.017]])

    Q1 = M1.T @ M1
    Q2 = M2.T @ M2
    W1 = M1.T @ S
    W2 = M2.T @ S

    # Sets up the two states of the world
    v1 = world(A=A, B=B, C=C, R=R, Q=Q1, W=W1)
    v2 = world(A=A, B=B, C=C, R=R, Q=Q2, W=W2)

    MJLQBarro = LQ_Markov(β, Π, v1, v2)

The decision rules are now dependent on the Markov state:

.. code-block:: python3

    MJLQBarro.F[1]

.. code-block:: python3

    MJLQBarro.F[2]

Simulating a large number of such economies over time reveals
interesting dynamics

Debt tends to stay low and stable, but
periodically spikes up to high levels

.. code-block:: python3

    T = 2000
    x0 = np.array([[1000, 1, 25]])
    for i in range(250):
        x, u, w, s = MJLQBarro.compute_sequence(x0, ts_length=T)
        plt.plot(list(range(T+1)), x[0, :])
    plt.xlabel('Time')
    plt.ylabel('Taxation')
    plt.show()
