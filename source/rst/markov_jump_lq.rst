.. _markov_jump_lq:

.. include:: /_static/includes/lecture_howto_py.raw

.. index::
    single: python


************************************************
Markov Jump Linear Quadratic Dynamic Programming
************************************************

.. contents:: :depth: 2

**Co-author:** `Sebastian Graves <https://github.com/sebgraves>`__


Overview
=========

This lecture describes infinite-horizon **Markov jump linear quadratic dynamic programming**, a powerful extension of the method described in
:doc:`LQ dynamic programming <lqcontrol>` 


**Markov jump linear quadratic dynamic programming** combines advantages
of

-  the computational simplicity of **linear quadratic dynamic
   programming**, with

-  the ability of **finite state Markov chains** to represent
   interesting patterns of random variation

The idea is to replace the constant matrices that define a **linear quadratic
dynamic programming problem** with :math:`N` sets of matrices that are fixed functions of
the state of an :math:`N` state Markov chain

For the class of infinite horizon problems being studied in this lecture, this leads to :math:`N` interrelated
matrix Riccati equations that determine :math:`N` optimal value
functions and :math:`N` linear decision rules

One of these value functions and one of these decision rules apply in each of the :math:`N` Markov states

That is,  when the Markov state is in state :math:`j`, the value function and the decision rule
for state :math:`j` prevails


Review of useful LQ dynamic programming formulas
=================================================

To begin,it is handy to have the following reminder in mind

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


   P = R+ \beta A' P A
             -(\beta B'  P A + W)' (Q + \beta B P B )^{-1} (\beta B P A + W)

and the constant :math:`\rho` satisfies

.. math::

   \rho = \beta
     \left( \rho + {\rm trace}(P C C') \right)

and the matrix :math:`F` in the decision rule for :math:`u_t` satisfies

.. math::


   F = (Q + \beta  B' P B)^{-1} (\beta (B' P A )+ W)


With the preceding formulas in mind, we are ready to approach Markov Jump LQ dynamic programming


Linked Ricatti equations for Markov LQ dynamic programming
===========================================================

The key idea is to make the matrices :math:`A, B, C, R, Q, W` fixed
functions of a finite state :math:`s` that is governed by an :math:`N`
state Markov chain

This makes decision rules depend on the Markov
state, and so fluctuate through time in limited ways

In particular, we use the following extension of a discrete time linear
quadratic dynamic programming problem

We let :math:`s(t) \equiv s_t \in [1, 2, \ldots, N]` be a time :math:`t` realization of an
:math:`N`-state Markov chain with transition matrix :math:`\Pi` having
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



Applications
==============

Zejin and Tom will add some simple ones here from Tom's *.pdf notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


More examples
^^^^^^^^^^^^^^

In the following  lectures, we describe how Markov jump  linear quadratic dynamic programming  can be used to extend the :cite:`Barro1979` model
of optimal tax-smoothing and government debt in particular directions

  1. :doc:`How to Pay for a War: Part 1 <tax_smoothing_1>`

  2. :doc:`How to Pay for a War: Part 2 <tax_smoothing_2>`

  3. :doc:`How to Pay for a War: Part 3 <tax_smoothing_3>`