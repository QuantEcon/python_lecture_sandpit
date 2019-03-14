.. _HS_recursive_models:

.. include:: /_static/includes/lecture_howto_py.raw

.. index::
    single: python



***************************************************
Recursive Models of Dynamic Linear Economies
***************************************************
 
.. contents:: :depth: 2 

.. epigraph::

    "Complete market economies are all alike" --     Robert E. Lucas, Jr., (1989)


.. epigraph::

    "Every partial equilibrium model can be interpreted as a general equilibrium model." --   Anonymous


Diverse Models
==============

-  Lucas asset pricing model.

-  Lucas-Prescott model of investment under uncertainty.

-  Asset pricing models with habit persistence.

-  Rosen-Topel equilibrium model of housing.

-  Rosen schooling models.

-  Rosen-Murphy-Scheinkman model of cattle cycles.

-  Hansen-Sargent-Tallarini model of robustness and asset pricing.

-  Many more :math:`\ldots`

.. _section-1:

-  Their apparent diversity conceals an essential unity

-  All are special cases of a linear-quadratic-Gaussian model of general
   economic equilibrium.

.. _section-2:

  

Complete Markets Economies
============================

Common objects and features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Commodity space.

-  Technology.

-  People and their preferences over commodities.

-  Price system.

-  One budget constraint per person.

-  Equilibrium definition.

-  Two welfare theorems.

-  Presence of a representative consumer.

Absence of Frictions Such as :math:`\ldots`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Enforcement.

-  Information asymmetries.

-  Other forms of transactions costs.

-  Externalities.

Hicks-Arrow tricks
^^^^^^^^^^^^^^^^^^^^

Imperialism of complete markets models comes from:

-  Indexing commodities and their prices by time (Hicks).

-  Indexing commodities and their prices by chance (Arrow).

Forecasting?
^^^^^^^^^^^^^

-  Consequence of single budget constraint plus Hicks-Arrow tricks:
   households and firms need not forecast.

-  But there exist equivalent structures – recursive competitive
   equilibria – where they do appear to need to forecast. To forecast,
   they use: (a) equilibrium pricing functions, and (b) knowledge of the
   Markov structure of the economy’s state vector.

Theory and Econometrics
^^^^^^^^^^^^^^^^^^^^^^^^

-  Outcome of theorizing is a stochastic process, i.e., a probability
   distribution over sequences of prices and quantities, indexed by
   parameters describing preferences, technologies, and information
   flows.

-  Another name for that object is a likelihood function, a central
   object of both frequentist and Bayesian statistics.

A Class of Economies
====================

Basic Ideas:
^^^^^^^^^^^^^

-  An economy consists of a list of matrices that describe peoples’
   household technologies, their preferences over consumption services,
   their production technologies, and their information sets.

-  Complete markets.

-  Competitive equilibrium allocations and prices satisfy equations that
   are easy to write down and solve.

-  Competitive equilibrium outcomes have representations that are
   convenient econometrically.

-  Different example economies manifest themselves simply as different
   settings for various matrices.

Tools
^^^^^^

-  A theory of recursive dynamic competitive economies;

-  Linear optimal control theory;

-  Recursive methods for estimating and interpreting vector
   autoregressions;

-  Python, Julia, MATLAB.

Representative Household
--------------------------

Alternative meanings:

-  A single ‘stand-in’ household (Prescott).

-  Heterogeneous households satisfying conditions for Gorman aggregation
   into a representative household.

-  Heterogeneous household technologies violating conditions for Gorman
   aggregation but susceptible to aggregation into a single
   representative household via ‘non-Gorman aggregation’.

.. _representative-household-1:



Remark: There is a sense in which a representative agent exists for any
complete markets economy (‘mongrel’ or ‘non-Gorman’ aggregation)

.. _representative-household-2:



-  The three alternative senses have different consequences in terms how
   prices and allocations can be computed.

-  Can prices and an aggregate allocation be computed before the
   allocation to individual heterogeneous households?

-  Answers are “Yes” for Gorman aggregation, “No” for non-Gorman
   aggregation.

Insights and Practical Benefits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Deeper understanding comes from recognizing common underlying
   structures

-  Unleash a common suite of programs (Python).

Mathematical Tools
-------------------

Duality

-  Stochastic Difference Equations (Linear).

-  LQ Dynamic Programming and Linear Filtering (they are the same thing
   mathematically).

-  Spectral Factorization Identity (for understanding vector
   autoregressions and non-Gorman aggregation).

Roadmap
--------

-  Information.

-  Technology.

-  Preferences.

-  Equilibrium concept and computation.

-  Econometric representation and estimation.

.. _section-3:

Stochastic Model of Information Flows and Outcomes
====================================================

Linear Stochastic Difference Equations
-----------------------------------------

The sequence :math:`\{w_t : t=1,2, \ldots\}` is said to be a martingale
difference sequence adapted to :math:`\{J_t : t=0, 1, \ldots \}` if
:math:`E(w_{t+1} \vert J_t) = 0` for :math:`t=0, 1, \ldots\,`.

The sequence :math:`\{w_t : t=1,2,\ldots\}` is said to be conditionally
homoskedastic if :math:`E(w_{t+1}w_{t+1}^\prime \mid J_t) = I` for
:math:`t=0,1, \ldots\,`.

We assume that the :math:`\{w_t : t=1,2,\ldots\}` process is
conditionally homoskedastic.

.. _linear-stochastic-difference-equations-1:



Let :math:`\{x_t : t=1,2,\ldots\}` be a sequence of
:math:`n`-dimensional random vectors, i.e. an :math:`n`-dimensional
stochastic process.


The process :math:`\{x_t : t=1,2,\ldots\}` is constructed recursively
using an initial random vector
:math:`x_0\sim {\mathcal N}(\hat x_0, \Sigma_0)` and a time-invariant
law of motion:

.. math:: 

    x_{t+1} = Ax_t + Cw_{t+1}
    
for :math:`t=0,1,\ldots`  where :math:`A` is an :math:`n` by :math:`n` matrix and :math:`C` is an
:math:`n` by :math:`N` matrix.


Evidently, the distribution of :math:`x_{t+1}` given :math:`x_t` is
:math:`{\mathcal N}(Ax_t, CC')`

Information Sets
-----------------

Let :math:`J_0` be generated by :math:`x_0` and :math:`J_t` be generated
by :math:`x_0, w_1, \ldots ,
w_t`, which means that :math:`J_t` consists of the set of all measurable
functions of :math:`\{x_0, w_1,\ldots,
w_t\}`

Prediction Theory
------------------

The optimal forecast of :math:`x_{t+1}` given current information is

.. math::

    E(x_{t+1} \mid J_t) = Ax_t 

 and the one-step-ahead forecast error is

.. math:: 

    x_{t+1} - E(x_{t+1} \mid J_t) = Cw_{t+1} 

 The covariance matrix of :math:`x_{t+1}` conditioned on :math:`J_t` is

 

.. math::

    E (x_{t+1} - E ( x_{t+1} \mid J_t) ) (x_{t+1} - E ( x_{t+1} \mid J_t))^\prime  = CC^\prime 

A nonrecursive expression for :math:`x_t` as a function of
:math:`x_0, w_1, w_2, \ldots,  w_t` is

.. math::

   \begin{eqnarray*}
    x_t & = & Ax_{t-1} + Cw_t \cr
   & = & A^2 x_{t-2} + ACw_{t-1} + Cw_t \cr
   & =  & \Bigl[\sum_{\tau=0}^{t-1} A^\tau Cw_{t-\tau} \Bigr] + A^t x_0 . \end{eqnarray*}

.. _prediction-theory-1:


Shift forward in time:

.. math:: 

    x_{t+j} = \sum^{j-1}_{s=0} A^s C w_{t+j-s} + A^j x_t 

Projecting on the information set :math:`\{ x_0, w_t, w_{t-1},
\ldots, w_1\}` gives

.. math:: E_t x_{t+j} = A^j x_t 

where :math:`E_t (\cdot) \equiv  E [ (\cdot) \mid x_0, w_t, w_{t-1}, \ldots, w_1]
= E (\cdot) \mid J_t`, and :math:`x_t` is in :math:`J_t`.

.. _prediction-theory-2:




It is useful to obtain the covariance matrix of the :math:`j`-step-ahead
prediction error :math:`x_{t+j} - E_t x_{t+j} = \sum^{j-1}_{s=0} A^s C w_{t-s+j}`

Evidently,

.. math::

   E_t (x_{t+j} - E_t x_{t+j})  (x_{t+j} - E_t x_{t+j})^\prime =
   \sum^{j-1}_{k=0} A^k C C^\prime A^{k^\prime} \equiv v_j .

:math:`v_j` can be calculated recursively via

.. math::

   \begin{eqnarray*}
    v_1 &= & CC^\prime \cr
    v_j &=  & CC^\prime + A v_{j-1} A^\prime, \quad j \geq 2 . \end{eqnarray*}

Orthogonal Decomposition
========================

To decompose these covariances into parts attributable to the individual
components of :math:`w_t`, we let :math:`i_\tau` be an
:math:`N`-dimensional column vector of zeroes except in position
:math:`\tau`, where there is a one. Define a matrix
:math:`\upsilon_{j,\tau}`

.. math::

   \upsilon_{j,\tau} = \sum_{k=0}^{j-1} A^k C i_\tau i_\tau^\prime C^\prime
   A^{^\prime k} .

Note that :math:`\sum_{\tau=1}^N i_\tau i_\tau^\prime = I`, so that we
have

.. math:: \sum_{\tau=1}^N \upsilon_{j, \tau} = \upsilon_j .

Evidently, the matrices
:math:`\{ \upsilon_{j, \tau} , \tau = 1, \ldots, N \}` give an
orthogonal decomposition of the covariance matrix of
:math:`j`-step-ahead prediction errors into the parts attributable to
each of the components :math:`\tau =
1, \ldots, N`.

Taste and Technology Shocks
----------------------------

:math:`E(w_t \mid J_{t-1}) = 0` and :math:`E(w_t
w_t^\prime \mid J_{t-1}) = I` for :math:`t=1,2, \ldots`

.. math:: 

    b_t = U_b z_t \hbox{ and } d_t = U_dz_t,

:math:`U_b` and :math:`U_d` are matrices that select entries of
:math:`z_t`. The law of motion for :math:`\{z_t : t=0, 1, \ldots\}` is

.. math:: 

    z_{t+1} = A_{22} z_t + C_2 w_{t+1} \ \hbox { for } t = 0, 1, \ldots .

where :math:`z_0` is a given initial condition. The eigenvalues of the
matrix :math:`A_{22}` have absolute values that are less than or equal
to one.

Other Components of Economies
=============================

-  Production technologies.

-  Household technologies.

-  Preferences.

Summary
=======

Information and shocks:
^^^^^^^^^^^^^^^^^^^^^^^
.. math::

   \begin{eqnarray*}
    z_{t+1} & = &A_{22} z_t + C_2 w_{t+1}
   \cr  b_t & = & U_b z_t \cr d_t & = & U_d z_t .\end{eqnarray*}

Production Technology:
^^^^^^^^^^^^^^^^^^^^^^^^
.. math::

   \begin{eqnarray*}
    \Phi_c c_t +  \Phi_g g_t + \Phi_i i_t &= &\Gamma k_{t-1} + d_t \cr
   k_t &= &\Delta_k k_{t-1} + \Theta_k i_t \cr g_t \cdot g_t & = &\ell_t^2 . \end{eqnarray*}

Household Technology:
^^^^^^^^^^^^^^^^^^^^^^
.. math::

   \begin{eqnarray*}
    \cr s_t & = &
   \Lambda h_{t-1} + \Pi c_t \cr h_t & = & \Delta_h h_{t-1} + \Theta_h c_t \end{eqnarray*}

Preferences:
^^^^^^^^^^^^^

.. math::

   \Bigl( {1 \over 2}\Bigr)  E \sum_{t=0}^\infty \beta^t [ (s_t -
   b_t) \cdot ( s_t - b_t) + \ell_t^2 ] \bigl| J_0 , \ 0 < \beta < 1

.. _summary-1:


Production Technologies
-------------------------

.. math:: \Phi_c c_t + \Phi_g g_t + \Phi_i i_t = \Gamma k_{t-1} + d_t .

.. math:: \mid g_t \mid \leq \ell_t



Assumption: :math:`[\Phi_c\ \Phi_g]` is nonsingular.

Endowment Economy
------------------

There is a single consumption good that cannot be stored over time. In
time period :math:`t`, there is an endowment :math:`d_t` of this single
good. There is neither a capital stock, nor an intermediate good, nor a
rate of investment. So :math:`c_t = d_t`.

To implement this specification, We can choose :math:`A_{22}, C_2`, and
:math:`U_d` to make :math:`d_t` follow any of a variety of stochastic
processes. To satisfy our earlier rank assumption, we set:

.. math:: 

    c_t + i_t = d_{1t}

.. math::

    g_t = \phi_1 i_t

where :math:`\phi_1` is a small positive number.

To implement this
version, we set :math:`\Delta_k = \Theta_k = 0` and

.. math::

   \Phi_c =  \begin{bmatrix}  1 \cr 0 \cr \end{bmatrix},
   \Phi_i = \begin{bmatrix} 1 \cr \phi_1 \cr \end{bmatrix} , \ \ \Phi_g =
   \begin{bmatrix} 0 \cr -1 \cr \end{bmatrix},  \ \ \Gamma = \begin{bmatrix}
    0 \cr 0 \end{bmatrix},  \ \ d_t  = \begin{bmatrix} d_{1t}
   \cr 0 \end{bmatrix} .

We can use this specification to create a linear-quadratic version of
Lucas’s (1978) asset pricing model.

Single-Period Adjustment Costs
==============================

There is a single consumption good, a single intermediate good, and a
single investment good. The technology obeys

.. math::

   \begin{eqnarray*}
   c_t &=  &\gamma k_{t-1} + d_{1t} ,\ \ \gamma > 0 \cr
   \phi_1 i_t &= & g_t + d_{2t}, \ \ \phi_1 > 0 \cr
   \ell^2_t &= &  g^2_t \cr
   k_t &= & \delta_k k_{t-1} + i_t ,\ 0< \delta_k < 1 \end{eqnarray*}

Set

.. math::

   \Phi_c = \begin{bmatrix}1 \cr 0 \end{bmatrix} ,\ \Phi_g = \begin{bmatrix}0 \cr
   -1 \end{bmatrix}, \ \Phi_i = \begin{bmatrix} 0 \cr \phi_1 \end{bmatrix}

.. math::

   \Gamma = \begin{bmatrix} \gamma \cr 0 \end{bmatrix}, \ \Delta_k = \delta_k,
   \ \Theta_k = 1 .

We set :math:`A_{22}, C_2` and :math:`U_d` to make
:math:`(d_{1t}, d_{2t})^\prime = d_t` follow a desired stochastic
process.

Preferences
===========

Preferences of a Representative Household:

.. math::

   -\left({1 \over 2}\right) E \sum^\infty_{t=0} \beta^t \left[ (s_t - b_t) \cdot (s_t -
   b_t) + (\ell_t)^2 \right] \mid J_0 \quad ,\ 0 < \beta < 1

Household Technologies 
=======================

.. math:: h_t = \Delta_h h_{t-1} + \Theta_h c_t

.. math:: s_t = \Lambda h_{t-1} + \Pi c_t .

Assumption: The absolute values of the eigenvalues of :math:`\Delta_h`
are less than or equal to one.



Canonical Household Technologies:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

They satisfy an ‘invertibility’
requirement relating sequences :math:`\{s_t\}` of services and
:math:`\{c_t\}` of consumption flows.

.. raw:: latex

   \bigskip

**Note:** Later we’ll describe how to obtain a canonical representation
of a household technology from one that is not canonical.

Time Separability
------------------

.. math::

   -{1\over 2} E \sum^\infty_{t=0} \beta^t \left[ (c_t - b_t)^2 + \ell_t^2
   \right] \mid J_0 \quad ,\ 0 < \beta < 1

Consumer Durables
------------------

.. math:: h_t = \delta_h h_{t-1} + c_t \quad ,\ 0 < \delta_h < 1 .

Services at :math:`t` are related to the stock of durables at the
beginning of the period:

.. math:: s_t = \lambda h_{t-1} \ , \ \lambda > 0  .

Preferences are ordered by

.. math::

   -{1 \over 2} E \sum^\infty_{t=0} \beta^t \left[(\lambda h_{t-1} -
   b_t)^2 + \ell_t^2\right] \mid J_0

Set :math:`\Delta_h = \delta_h,
\Theta_h =1, \Lambda = \lambda, \Pi = 0`.

Habit Persistence
-------------------

.. math::

   -\Bigl({1\over 2}\Bigr)\, E \sum^\infty_{t=0} \beta^t \Bigl[\bigl(c_t - \lambda
    (1-\delta_h) \sum^\infty_{j=0}\, \delta^j_h\, c_{t-j-1}-b_t\bigr)^2+\ell^2_t\Bigl]  \bigl| J_0

.. math::

    0<\beta < 1\ ,\ 0 < \delta_h < 1\ ,\ \lambda > 0 

Here the effective bliss point :math:`b_t + \lambda (1 - \delta_h)
\sum^\infty_{j=0} \delta^j_h\, c_{t-j-1}` shifts in response to a moving
average of past consumption.


Preferences of this form require an initial condition for the geometric
sum :math:`\sum^\infty_{j=0} \delta_h^j c_{t - j-1}` that we specify as
an initial condition for the ‘stock of household durables,’
:math:`h_{-1}`.

.. _habit-persistence-1:

Habit Persistence
-------------------

Set

.. math:: h_t = \delta_h h_{t-1} + (1-\delta_h) c_t \quad ,\ 0 < \delta_h < 1 .

.. math::

   h_t = (1 - \delta_h) \sum^t_{j=0} \delta_h^j\, c_{t-j} + \delta^{t+1}_h\,
   h_{-1} .

.. math:: s_t = - \lambda h_{t-1} + c_t, \ \lambda > 0 .

To implement, set
:math:`\Lambda = -\lambda,\ \Pi = 1,\ \Delta_h = \delta_h,\ \Theta_h=1-\delta_h`.

Seasonal Habit Persistence
---------------------------

.. math::

   -\Bigl({1\over 2}\Bigr) \, E \sum^\infty_{t=0} \beta^t  \Bigl[\bigl(c_t - \lambda
    (1-\delta_h) \sum^\infty_{j=0}\, \delta^{j}_h\, c_{t-4j-4}-b_t\bigr)^2+\ell^2_t\Bigr]

.. math:: 

    0<\beta < 1\ ,\ 0 < \delta_h < 1\ ,\ \lambda > 0 

Here the effective bliss point :math:`b_t + \lambda (1 - \delta_h) \sum^\infty_{j=0} \delta^j_h\, c_{t-4j-4}` shifts in response to a
moving average of past consumptions of the same quarter.

.. _seasonal-habit-persistence-1:


To implement, set

.. math:: 

    \tilde h_t = \delta_h \tilde h_{t-4} + (1-\delta_h) c_t \quad ,\ 0 < \delta_h < 1 

This implies that

.. math::

    h_t = \begin{bmatrix}\tilde h_t \cr
          \tilde h_{t-1}\cr
          \tilde h_{t-2}\cr
          \tilde  h_{t-3}\end{bmatrix}  =
          \begin{bmatrix} 0 & 0 & 0 & \delta_h \cr
                    1 & 0 & 0 & 0 \cr
                    0 & 1 & 0 & 0 \cr
                    0 & 0 & 1 & 0 \end{bmatrix}
                    \begin{bmatrix} \tilde h_{t-1} \cr \tilde h_{t-2} \cr \tilde h_{t-3} \cr \tilde h_{t-4} \end{bmatrix}
                    + \begin{bmatrix}(1 - \delta_h) \cr 0 \cr 0 \cr 0 \end{bmatrix} c_t

with consumption services

.. math:: 

    s_t = - \begin{bmatrix}0 & 0 & 0 & -\lambda\end{bmatrix}  h_{t-1} + c_t \quad , \ \lambda > 0 .

Adjustment Costs
-----------------

.. math::

   -\Bigl({1 \over 2}\Bigr) E \sum^\infty_{t=0} \beta^t [(c_t - b_{1t})^2 +
   \lambda^2 (c_t - c_{t-1})^2 + \ell^2_t ] \mid J_0

.. math:: 0 < \beta < 1 \quad, \ \lambda > 0 ,

.. _adjustment-costs-1:

Adjustment Costs
================

To capture, set

.. math:: h_t  = c_t

.. math::

   s_t = \begin{bmatrix} 0 \cr - \lambda \end{bmatrix} h_{t-1} +
   \begin{bmatrix} 1 \cr \lambda \end{bmatrix} c_t

 so that

.. math:: s_{1t} = c_t

.. math:: s_{2t} = \lambda (c_t - c_{t-1} ) .

 We set the first component :math:`b_{1t}` of :math:`b_t` to capture the
stochastic bliss process, and set the second component identically equal
to zero. Thus, we set :math:`\Delta_h = 0, \Theta_h = 1`,

.. math::

   \Lambda = \begin{bmatrix} 0 \cr -\lambda \end{bmatrix}\ ,\ \Pi =
   \begin{bmatrix} 1 \cr \lambda \end{bmatrix} .

Multiple Consumption Goods
==========================

.. math::

   \Lambda = \begin{bmatrix} 0\cr0\end{bmatrix} \ \hbox { and } \ \Pi =
   \begin{bmatrix}\pi_1 & 0 \cr \pi_2 & \pi_3 \end{bmatrix} .

.. math:: -{1 \over 2} \beta^t (\Pi c_t - b_t)^\prime (\Pi c_t - b_t).

.. math:: mu_t= - \beta^t [\Pi^\prime \Pi\, c_t - \Pi^\prime\, b_t].

.. math::

   c_t = - (\Pi^\prime \Pi)^{-1} \beta^{-t} mu_t + (\Pi^\prime \Pi)^{-1}
   \Pi^\prime b_t .

This is called the Frisch demand function for consumption.


We can think of the vector :math:`mu_t` as playing the role of prices,
up to a common factor, for all dates and states. The scale factor is
determined by the choice of numeraire.

Substitutes and Complements
----------------------------

Notions of substitutes and complements can be defined in terms of these
Frisch demand functions. Two goods can be said to be substitutes if the
cross-price effect is positive and to be complements if this effect is
negative. Hence this classification is determined by the off-diagonal
element of :math:`-(\Pi^\prime \Pi)^{-1}`, which is equal to
:math:`\pi_2 \pi_3 /\det
(\Pi^\prime \Pi)`. If :math:`\pi_2` and :math:`\pi_3` have the same
sign, the goods are substitutes. If they have opposite signs, the goods
are complements.

Square Summability
-------------------

Impose:

.. math::

   E \sum^\infty_{t=0} \beta^t h_t \cdot h_t \mid J_0 < \infty\ \hbox { and }\
   E \sum^\infty_{t=0} \beta^t k_t \cdot k_t \mid J_0 < \infty .

Define:

.. math::

   L_0^2 = [ \{ y_t \}  : y_t \ \hbox{is a random
   variable in } \  J_t \ \hbox{ and } \  E \sum_{t=0}^\infty \beta^t
   y_t^2 \mid J_0 < + \infty] .

Require that each component of :math:`h_t` and each component of
:math:`k_t` belong to :math:`L_0^2`.

.. _summary-2:

Summary
--------

Information and shocks:

.. math::

   \begin{eqnarray*}
    z_{t+1} & = &A_{22} z_t + C_2 w_{t+1}
   \cr  b_t & = & U_b z_t \cr d_t & = & U_d z_t .\end{eqnarray*}

 Production Technology:

.. math::

   \begin{eqnarray*}
    \Phi_c c_t &+ & \Phi_g g_t + \Phi_i i_t = \Gamma k_{t-1} + d_t \cr
   k_t &= &\Delta_k k_{t-1} + \Theta_k i_t \cr g_t \cdot g_t & = &\ell_t^2 . \end{eqnarray*}

 Household Technology:

.. math::

   \begin{eqnarray*}
    \cr s_t & = &
   \Lambda h_{t-1} + \Pi c_t \cr h_t & = & \Delta_h h_{t-1} + \Theta_h c_t \end{eqnarray*}

 Preferences:

.. math::

   \Bigl( {1 \over 2}\Bigr)  E \sum_{t=0}^\infty \beta^t [ (s_t -
   b_t) \cdot ( s_t - b_t) + \ell_t^2 ] \bigl| J_0 , \ 0 < \beta < 1

.. _summary-3:

Summary
=======

Information and shocks:

.. math::

   \begin{eqnarray*}
    z_{t+1} & = &\{A_{22}} z_t + \{C_2} w_{t+1}
   \cr  b_t & = & \{U_b} z_t \cr d_t & = & \{U_d} z_t .\end{eqnarray*}

 Production Technology:

.. math::

   \begin{eqnarray*}
    \{\Phi_c} c_t +  \{\Phi_g} g_t + \{\Phi_i} i_t& = &\{\Gamma} k_{t-1} + d_t \cr
   k_t &= &\{\Delta_k} k_{t-1} + \{\Theta_k} i_t \cr g_t \cdot g_t & = &\ell_t^2 . \end{eqnarray*}

Household Technology:

.. math::

   \begin{eqnarray*}
    \cr s_t & = &
   \{\Lambda} h_{t-1} + \{\Pi} c_t \cr h_t & = & \{\Delta_h} h_{t-1} + \{\Theta_h} c_t \end{eqnarray*}

Preferences:

.. math::

   \Bigl( {1 \over 2}\Bigr)  E \sum_{t=0}^\infty \{\beta}^t [ (s_t -
   b_t) \cdot ( s_t - b_t) + \ell_t^2 ] \bigl| J_0 , \ 0 < \beta < 1

Battle Plan

-  Planning Problem

-  Competitive Equilibrium


Optimal Resource Allocations
============================

Choose :math:`\{c_t, i_t, g_t\}_{t=0}^\infty` to maximize

.. math::

   -(1/2)E \sum_{t=0}^\infty \beta^t [ (s_t - b_t) \cdot (s_t - b_t) + g_t
   \cdot g_t ] \bigl| J_0 .

 subject to

.. math::

   \begin{eqnarray*}
    \Phi_c c_t &+ & \Phi_g \, g_t + \Phi_i i_t = \Gamma k_{t-1} + d_t,
   \cr
   k_t & = & \Delta_k k_{t-1} + \Theta_k i_t , \cr
   h_t & = & \Delta_h h_{t-1} + \Theta_h c_t , \cr
   s_t & =  &\Lambda h_{t-1} + \Pi c_t , \cr
     z _{t+1} & = & A_{22} z_t + C_2 w_{t+1} , \ b_t = U_b z_t,  \  \hbox{ and } \
   d_t = U_d z_t \end{eqnarray*}

 :math:`h_{-1},
k_{-1}`, and :math:`z_0` are initial conditions.

Two Formulations

-  Lagrangian

-  Dynamic Programming

Lagrangian Formulation
======================

.. math::

   \begin{eqnarray*}
    {\mathcal L} &= & - E \sum_{t=0}^\infty \beta^t \biggl[
   \Bigl( {1 \over 2} \Bigr) [ (s_t - b_t) \cdot (s_t - b_t) + g_t
   \cdot g_t] \cr &+ &   {\cal M}_t^{d \prime} \cdot ( \Phi _cc_t  +
   \Phi_gg_t + \Phi_ii_t - \Gamma k_{t-1} - d_t ) \cr &+ &{\cal M}_t^{k
   \prime} \cdot (k_t - \Delta_k k_{t-1} - \Theta_k i_t ) \cr &+ & {\cal
   M}_t^{h \prime} \cdot (h_t - \Delta_h h_{t-1} - \Theta_h c_t) \cr &+ &
   {\cal M}_t^{s \prime} \cdot (s_t - \Lambda h_{t-1} - \Pi c_t )
   \biggr] \Bigl| J_0 . \end{eqnarray*}

FONC’s 
=======

First-order necessary conditions for maximization with respect to
:math:`c_t, g_t,
h_t, i_t, k_t`, and :math:`s_t`, respectively, are:

.. math::

   \begin{eqnarray*}
    - \Phi_c^\prime  {\cal M}_t^d &+  &\Theta_h^\prime {\cal
   M}_t^h + \Pi^\prime {\cal M}_t^s = 0 , \cr
   & - & g_t - \Phi_g^\prime  {\cal M}_t^d = 0 , \cr
   - {\cal M}_t^h &+ & \beta E ( \Delta_h^\prime {\cal M}^h_{t+1} +
   \Lambda^\prime {\cal M}_{t+1}^s ) \mid J_t = 0 , \cr
   & - & \Phi_i^\prime {\cal M}_t^d + \Theta_k^\prime {\cal M}_t^k = 0 , \cr
   - {\cal M}_t^k &+& \beta E ( \Delta_k^\prime {\cal M}^k_{t+1} + \Gamma^\prime
   {\cal M}_{t+1}^d) \mid J_t = 0 , \cr
   & - & s_t + b_t - {\cal M}_t^s = 0 \end{eqnarray*}

 for :math:`t=0,1, \ldots`. In addition, we have the complementary
slackness conditions (these recover the original transition equations)
and the transversality conditions

.. math::

   \begin{eqnarray*}
    \lim_{t \to \infty}& \beta^t & E [ {\cal M}_t^{k \prime} k_t ]
   \mid J_0 = 0  \cr
    \lim_{t \to \infty}& \beta^t&  E [ {\cal M}_t^{h \prime} h_t ]
   \mid J_0 = 0. \end{eqnarray*}

Dynare
======

-  The system formed by the FONCs and the transition equations can be
   handed over to dynare.

-  Dynare will solve planning problem for fixed parameter values.

-  Dynare will allow you to estimate the free parameters by maximum
   likelihood or Bayes’ Law.

Dynare Ready Equations
======================

.. math::

   \begin{eqnarray*}
    - \Phi_c^\prime  {\cal M}_t^d &+  &\Theta_h^\prime {\cal
   M}_t^h + \Pi^\prime {\cal M}_t^s = 0 , \cr
   & - & g_t - \Phi_g^\prime  {\cal M}_t^d = 0 , \cr
   - {\cal M}_t^h &+ & \beta E ( \Delta_h^\prime {\cal M}^h_{t+1} +
   \Lambda^\prime {\cal M}_{t+1}^s ) \mid J_t = 0 , \cr
   & - & \Phi_i^\prime {\cal M}_t^d + \Theta_k^\prime {\cal M}_t^k = 0 , \cr
   - {\cal M}_t^k &+& \beta E ( \Delta_k^\prime {\cal M}^k_{t+1} + \Gamma^\prime
   {\cal M}_{t+1}^d) \mid J_t = 0 , \cr
   & - & s_t + b_t - {\cal M}_t^s = 0 \cr
   \Phi_c c_t &+ & \Phi_g \, g_t + \Phi_i i_t = \Gamma k_{t-1} + d_t,
   \cr
   k_t & = & \Delta_k k_{t-1} + \Theta_k i_t , \cr
   h_t & = & \Delta_h h_{t-1} + \Theta_h c_t , \cr
   s_t & =  &\Lambda h_{t-1} + \Pi c_t , \cr
     z _{t+1} & = & A_{22} z_t + C_2 w_{t+1} , \ b_t = U_b z_t,  \  \hbox{ and } \
   d_t = U_d z_t  \end{eqnarray*}

Shadow Prices
=============

.. math:: {\cal M}_t^s = b_t - s_t .

.. math::

   {\cal M}_t^h = E \biggl[ \sum_{\tau =1}^\infty \beta^\tau
   (\Delta_h^\prime)^{\tau - 1} \Lambda^\prime {\cal M}_{t+ \tau}^s
   \mid J_t\biggr] .

.. math::

   {\cal M}_t^d = \begin{bmatrix}\Phi_c^\prime\cr \Phi_g^\prime\cr\end{bmatrix}^{-1}\
   \begin{bmatrix} \Theta_h^\prime {\cal M}_t^h  + \Pi^\prime {\cal M}_t^s \cr
   -g_t \cr  \end{bmatrix}.

.. math::

   {\cal M}_t^k = E \biggl[\sum _{\tau=1}^\infty \beta^\tau (\Delta_k^\prime)^{\tau-1}
   \Gamma^\prime {\cal M}_{t+ \tau}^d \mid J_t\biggr] .

.. math::

   {\cal M}_t^i =
   \Theta _k^\prime {\cal M}_t^k .

**Computational trick:** Use dynamic programming to get recursive representations
   for quantities and shadow prices.

Dynamic Programming 
====================

.. math::

   V(x_0) = \max_{c_0, i_0, g_0} [ - .5 [ (s_0 - b_0) \cdot (s_0 - b_0) + g_0 \cdot g_0 ] + \beta  E V
   (x_1) ]

 subject to the linear constraints

.. math::

   \begin{eqnarray*}
    \Phi _cc_0 & + & \Phi_g g_0 + \Phi_ii_0 = \Gamma k_{-1} + d_0 ,\cr
   k_0 & =  & \Delta_k k_{-1} + \Theta_k i_0 , \cr
   h_0 & = & \Delta_h h_{-1} + \Theta_h c_0 , \cr
   s_0 & = & \Lambda h_{-1} + \Pi c_0 , \cr
    z_1 & = & A_{22} z_0 + C_2 w_1,\ b_0 = U_b z_0 \ \hbox{ and }\  d_0 =
   U_d z_0 \end{eqnarray*}

.. math:: V(x) = x' P x + \rho

.. _dynamic-programming-1:

Dynamic Programming
===================

Choose a contingency plan for :math:`\{x_{t+1}, u_t \}_{t=0}^\infty` to
maximize

.. math::

   - E \sum_{t=0}^\infty  \beta^t [ x_t^\prime  R x_t + u_t^\prime  Q
   u_t + 2 u_t^\prime  W x_t ], \ 0 < \beta < 1

 subject to

.. math:: x_{t+1} = A x_t + B u_t + C w_{t+1}, \ t \geq 0 ,

 where :math:`x_0` is given; :math:`x_t` is an :math:`n \times 1` vector
of state variables, and :math:`u_t` is a :math:`k \times 1` vector of
control variables. We assume :math:`w_{t+1}` is a martingale difference
sequence with :math:`E w_t w_t^\prime = I`, and that :math:`C` is a
matrix conformable to :math:`x` and :math:`w`.

.. _dynamic-programming-2:

Dynamic Programming
===================

.. math::

   V (x_t) = \max_{u_t} \Bigl\{-( x_t^\prime R x_t + u_t^\prime Q u_t + 2
   u_t^\prime W x_t) + \beta E_t V (x_{t+1}) \Bigr\}

 subject to

.. math:: x_{t+1} = A x_t + B u_t + C w_{t+1}, \ t \geq 0 .

.. math:: V(x_t) = - x_t^\prime P x_t - \rho ,

 :math:`P` satisfies

.. math::

   P =  R + \beta A^\prime P A - (\beta A^\prime P
   B + W)   (Q + \beta B^\prime P B)^{-1} (\beta B^\prime P
   A + W') .

 This equation in :math:`P` is called the algebraic matrix Riccati
equation.

Decision rule: :math:`u_t = - F x_t`, where

.. math::

   F = (Q + \beta B^\prime P B)^{-1} (\beta B^\prime P A +
   W') .

 The optimum decision rule for :math:`u_t` is independent of the
parameters :math:`C`, and so of the noise statistics.

.. _dynamic-programming-3:

Dynamic Programming
===================

.. math::

   V_{j+1} (x_t) = \max_{u_t} \Bigl\{-( x_t^\prime R x_t + u_t^\prime Q
   u_t + 2 u_t^\prime W x_t) + \beta E_t V_j (x_{t+1}) \Bigr\}

.. math:: V_j (x_t) =- x_t^\prime P_{j} x_t - \rho_{j},

 where :math:`P_{j}` and :math:`\rho_{j}` satisfy the equations

.. math::

   \begin{eqnarray*}
    P_{j+1} & = & R + \beta A^\prime P_{j} A - (\beta
   A^\prime P_{j} B + W)  (Q + \beta B^\prime P_{j} B)^{-1} (\beta B^\prime P_{j}
   A + W')\cr  \rho_{j+1} & = &\beta \rho_{j} + \beta \ {\rm trace} \ P_{j} C C^\prime.
    \end{eqnarray*}

Planning as a Dynamic Programming Problem
=========================================

.. math::

   \max_{ \{u_t, x_{t+1}\} }\ - E \sum_{t=0}^\infty \beta^t [x_t^\prime
   Rx_t + u_t^\prime Q u_t + 2u_t^\prime Wx_t ] , \quad 0 < \beta < 1 ,

 subject to

.. math:: x_{t+1} = Ax_t + B u_t + Cw_{t+1} , \ t \geq 0

.. math::

   x_t = \begin{bmatrix} h_{t-1} \cr k_{t-1} \cr z_t \end{bmatrix} , \qquad
    u_t = i_t

Planning Problem as Dynamic Program
===================================

.. math::

   \begin{eqnarray*}
    A &= &\begin{bmatrix} \Delta_h & \Theta_h U_c [ \Phi_c \ \
   \Phi_g]^{-1} \Gamma & \Theta_h U_c [ \Phi_c \ \ \Phi_g]^{-1}  U_d \cr 0
   & \Delta_k & 0 \cr 0 & 0 & A_{22} \cr  \end{bmatrix} \cr \noalign{\smallskip}
   B &=& \begin{bmatrix} - \Theta_h U_c [ \Phi_c \ \ \Phi_g]^{-1} \Phi_i
   \cr \Theta_k \cr 0 \end{bmatrix}  \ ,\ C = \begin{bmatrix} 0 \cr 0 \cr
   C_2 \end{bmatrix} \end{eqnarray*}

.. math::

   \begin{bmatrix} x_t \cr u_t  \end{bmatrix}^\prime S \begin{bmatrix} x_t
   \cr u_t \end{bmatrix} = \begin{bmatrix} x_t \cr u_t \end{bmatrix}^\prime\ \
   \begin{bmatrix} R & W' \cr W & Q  \end{bmatrix}\ \ \begin{bmatrix} x_t
   \cr u_t \end{bmatrix}

 :math:`S = (G^\prime G + H^\prime H) / 2`

.. math::

   H = [\Lambda \ \vdots \ \Pi U_c [ \Phi_c \ \ \Phi_g]^{-1} \Gamma  \
   \vdots \ \Pi U_c [ \Phi_c \ \ \Phi_g]^{-1} U_d - U_b  \ \vdots \
    - \Pi U_c [\Phi_c \ \ \Phi_g]^{-1} \Phi_i]

.. math::

   G =
   U_g [ \Phi_c \ \ \Phi_g]^{-1} [0 \ \vdots \ \Gamma \ \vdots \ U_d \
   \vdots \ - \Phi_i] .

Lagrange Multipliers Equal Gradients of Planner’s Value Function
================================================================

.. math::

   \begin{eqnarray*}
   {\mathcal M}_t^k &= & M_k x_t\ \hbox{ and }\ {\cal M}_t^h = M_h
   x_t \ \hbox{ where } \cr
   M_k &= & 2 \beta [ 0 \ I \ 0 ] P A^o  \cr
   M_h &= & 2 \beta [ I \ 0 \ 0 ] P A^o . \end{eqnarray*}

.. math::

   {\mathcal M}_t^s = M_s x_t \ \hbox{ where }\ M_s = (S_b -
   S_s)\ \hbox{ and } \ S_b = [ 0 \ 0 \ U_b ] .

.. math::

   {\mathcal M}_t^d = M_d x_t\ \hbox{ where }\ M_d = \begin{bmatrix}
   \Phi_c^\prime \cr \Phi_g^\prime \cr \end{bmatrix} ^{-1}
   \begin{bmatrix}\Theta_h^\prime M_h + \Pi^\prime M_s \cr -S_g \cr \end{bmatrix}

.. math::

   {\mathcal M}_t^c = M_c x_t\ \hbox{ where }\ M_c = \Theta_h^\prime
   M_h + \Pi^\prime M_s

.. math:: {\mathcal M}_t^i = M_i x_t\ \hbox{ where } \ M_i = \Theta_k^\prime M_k .

Competitive Equilibrium
==========================


Commodity space
===============

.. math::

   L_0^2 = [ \{ y_t \}  : y_t \ \hbox{is a random
   variable in } \  J_t \ \hbox{ and } \  E \sum_{t=0}^\infty \beta^t
   y_t^2 \mid J_0 < + \infty]

Pricing Functional
==================

Values as Inner Products

.. math:: \pi (c) = E \sum_{t=0}^\infty \beta^t p_t^0 \cdot c_t \mid J_0 ,

 where :math:`p_t^0` belongs to :math:`L_0^2`.

.. _representative-household-3:

Representative Household
========================

Owns endowment process and initial stocks of :math:`h` and :math:`k`.
Chooses stochastic processes for :math:`\{c_t,\, s_t,\, h_t,\,
\ell_t\}^\infty_{t=0}`, each element of which is in :math:`L^2_0`, to
maximize

.. math::

   -\ {1 \over 2}\ E_0 \sum^\infty_{t=0} \beta^t\, \Bigl[(s_t-b_t) \cdot (s_t -
   b_t) + \ell_t^2\Bigr]

 subject to

.. math::

   E\sum^\infty_{t=0} \beta^t\, p^0_t \cdot c_t \mid J_0 = E \sum^\infty_{t=0}
   \beta^t\, (w^0_t \ell_t + \alpha^0_t \cdot d_t) \mid J_0 +
   v_0 \cdot k_{-1}

.. math:: s_t = \Lambda h_{t-1} + \Pi c_t

.. math::

   h_t = \Delta_h h_{t-1} + \Theta_h c_t, \quad h_{-1}, k_{-1}\
   \hbox{ given} .

Type I Firm
===========

A type I firm rents capital and labor and endowments and produces
:math:`c_t, i_t`. It chooses stochastic processes for
:math:`\{c_t, i_t, k_t, \ell_t,
g_t, d_t\}`, each element of which is in :math:`L^2_0`, to maximize

.. math::

   E_0\, \sum^\infty_{t=0} \beta^t\, (p^0_t \cdot c_t + q^0_t \cdot i_t - r^0_t
   \cdot k_{t-1} - w^0_t \ell_t - \alpha^0_t \cdot d_t)

 subject to

.. math:: \Phi_c\, c_t + \Phi_g\, g_t + \Phi_i\, i_t = \Gamma k_{t-1} + d_t

.. math:: -\, \ell_t^2 + g_t \cdot g_t = 0 .

Type II Firm
============

A firm of type II that acquires capital via investment and then rents
stocks of capital to the :math:`c,i`-producing type I firm. A type II
firm is a price taker facing the vector :math:`v_0` and the stochastic
processes :math:`\{r^0_t, q^0_t\}`. The firm chooses :math:`k_{-1}` and
stochastic processes for :math:`\{k_t, i_t\}^\infty_{t=0}` to maximize

.. math::

   E \sum^\infty_{t=0} \beta^t (r_t^0 \cdot k_{t-1} - q^0_t \cdot i_t) \mid
   J_0 - v_0 \cdot k_{-1}

 subject to

.. math:: k_t = \Delta_k k_{t-1} + \Theta_k i_t.

Competitive Equilibrium
=======================

A competitive equilibrium is a price system
:math:`[v_0, \{p^0_t, w^0_t, \alpha^0_t, q^0_t, r^0_t\}^\infty_{t=0}]`
and an allocation :math:`\{c_t, i_t, k_t, h_t, g_t, d_t\}^\infty_{t=0}`
that satisfy the following conditions:



Each component of the price system and the allocation resides in the
space :math:`L^2_0`.


Given the price system and given :math:`h_{-1},\, k_{-1}`, the
allocation solves the representative household’s problem and the
problems of the two types of firms.

Equilibrium Price System
========================

.. math::

   p^0_t = \bigl[\Pi^\prime {\cal M}^s_t + \Theta^\prime_h {\cal M}^h_t\bigr]/
   \mu^w_0 = {\cal M}^c_t / \mu^w_0

.. math:: w^0_t = \mid S_g x_t \mid / \mu^w_0

.. math:: r^0_t = \Gamma^\prime {\cal M}^d_t / \mu^w_0

.. math::

   q^0_t = \Theta^\prime_k {\cal M}^k_t / \mu^w_0 = {\cal M}^i_t /
   \mu^w_0

.. math:: \alpha^0_t =   {\cal M}^d_t / \mu^w_0

.. math::

   v_0 = \Gamma^\prime {\cal M}^d_0 / \mu^w_0 + \Delta^\prime_k
   {\cal M}^k_0 / \mu^w_0.

 With this price system, values can be assigned to the Lagrange
multipliers for each of our three classes of agents that cause all
first-order necessary conditions to be satisfied at these prices and at
the quantities associated with the optimum of the planning problem.

Asset Pricing
=============

Dividend Stream: :math:`\{y_t\} \in L^2_0`

Asset Value:
:math:`a_0 =  E\, \sum_{t=0}^\infty\, \beta^t\ p_t^0 \cdot y_t \mid J_0 .`

.. _asset-pricing-1:

Asset Pricing
=============

.. math:: y_t = U_a\, x_t

.. math:: a_0 = E \sum^\infty_{t=0}\, \beta^t\, x^\prime_t\, Z_a x_t \mid J_0

.. math:: Z_a = U^\prime_a M_c / \mu^w_0.

Convenient formulas:

.. math:: a_0 = x^\prime_0\, \mu_a\, x_0 + \sigma_a

.. math::

   \mu_a = \sum^\infty_{\tau=0}\, \beta^\tau\, (A^{o \prime})^\tau\ Z_a\,
   A^{o \tau}

.. math::

   \sigma_a = {\beta \over 1 - \beta}\ {\rm trace } \left( Z_a \sum^\infty_{\tau = 0}
   \,\beta^\tau\, (A^o)^\tau\, C C^\prime (A^{o \prime})^\tau \right).

Re-Opening Markets
==================

.. math::

   \begin{eqnarray*}
    L^2_t & = & [\{y_s\}^\infty_{s=t} : \ y_s \ \hbox{ is a random variable
   in }\ J_s\ \hbox{ for }\ s \geq t \cr
   & &\hbox {and } E\, \sum^\infty_{s=t}\, \beta^{s-t}\ y^2_s \mid J_t < + \infty] .\end{eqnarray*}

.. math:: p^t_s = M_c x_s / [\bar e_j M_c x_t ], \qquad s \geq t

.. math:: w^t_s = \mid S_g x_s \vert / [\bar e_j M_c x_t], \ \ s \geq t

.. math:: r^t_s = \Gamma^\prime M_d x_s / [\bar e_j M_c x_t],\ \ s \geq t

.. math:: q^t_s = M_i x_s / [\bar e_j \, M_c x_t], \qquad s \geq t

.. math:: \alpha^t_s =   M_d x_s / [\bar e_j \, M_c x_t] , \ \ s \geq t

.. math:: v_t = [\Gamma^\prime M_d + \Delta^\prime_k M_k] x_t / \, [\bar e_j \, M_c x_t]

Econometrics
==============

A tale of two state-space representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Facts Motivating Filtering
==========================

.. math:: H(y^t) \subset H(w^t,v^t)

.. math:: H(y^t) = H(a^t)

Statistical Representations
===========================

Original State-Space Representation:

.. math::

   \begin{eqnarray*}
    x_{t+1} & = & A^o x_t + Cw_{t+1} \cr
   y_t & =& Gx_t + v_t,  \end{eqnarray*}

where :math:`v_t` is a martingale difference sequence of measurement
errors that satisfies :math:`Ev_t
v_t' = R, E w_{t+1} v_s' = 0` for all :math:`t+1 \geq s` and

.. math:: x_0 \sim {\mathcal N}(\hat x_0,\Sigma_0).

 Innovations Representation:

.. math::

   \begin{eqnarray*}
   \hat x_{t+1} &= &A^o \hat x_t + K_t a_t \cr
   y_t &= & G \hat x_t + a_t,\end{eqnarray*}

 where
:math:`a_t = y_t - E[y_t | y^{t-1}], E a_t a_t^\prime \equiv \Omega_t =  G \Sigma_t G^\prime + R`.

.. _statistical-representations-1:

Statistical Representations
===========================

Original State-Space Representation:

.. math::

   \begin{eqnarray*}
    x_{t+1} & = & A^o x_t + C\color{red}{w_{t+1}} \cr
   y_t & =& Gx_t + \color{red}{v_t},  \end{eqnarray*}

\ where :math:`v_t` is a martingale difference sequence of measurement
errors that satisfies :math:`Ev_t
v_t' = R, E w_{t+1} v_s' = 0` for all :math:`t+1 \geq s` and

.. math:: x_0 \sim {\mathcal N}(\hat x_0,\Sigma_0).

 Innovations Representation:

.. math::

   \begin{eqnarray*}
   \hat x_{t+1} &= &A^o \hat x_t + K_t \color{red}{a_t} \cr
   y_t &= & G \hat x_t + \color{red}{a_t},\end{eqnarray*}

 where
:math:`a_t = y_t - E[y_t | y^{t-1}], E a_t a_t^\prime \equiv \Omega_t =  G \Sigma_t G^\prime + R`.

.. _facts-motivating-filtering-1:

Facts Motivating Filtering
==========================

.. math:: H(y^t) \subset H(w^t,v^t)

.. math:: H(y^t) = H(a^t)

Compare numbers of shocks in the two representations: :math:`n_w + n_y` versus  :math:`n_y` 

Compare spaces spanned

Kalman Filter
=============

.. math:: K_t = A^o \Sigma_t G^\prime (G \Sigma_t G^\prime + R)^{-1} .

 Riccati Difference Equation:

.. math::

   \begin{eqnarray*}
    \Sigma_{t+1} &= & A^o \Sigma_t A^{o \prime} + CC^\prime \cr
   &- & A^o \Sigma_t G^\prime (G \Sigma_t G^\prime + R)^{-1} G \Sigma_t A^{o \prime}. \end{eqnarray*}

Whitener
========

Whitening Filter:

.. math::

   \begin{eqnarray*}
    a_t & = &y_t - G \hat x_t \cr
   \hat x_{t+1} &= &  A^o \hat x_t + K_t  a_t \end{eqnarray*}

 can be used recursively to construct a record of innovations
:math:`\{ a_t \}^T_{t=0}` from an :math:`(\hat x_0, \Sigma_0)` and a
record of observations :math:`\{ y_t \}^T_{t=0}`.

Limiting Time-Invariant Innovations Representation
==================================================

.. math::

   \begin{eqnarray*}
    \Sigma & = & A^o \Sigma A^{o \prime} + CC^\prime \cr
   &- & A^o \Sigma G^\prime (G \Sigma G^\prime + R)^{-1} G \Sigma A^{o \prime} \cr
    K &= & A^o \Sigma_t G^\prime (G \Sigma G^\prime + R)^{-1}. \end{eqnarray*}

.. math::

   \begin{eqnarray*}
    \hat x_{t+1} &= & A^o \hat x_t + K a_t \cr
   y_t &= & G \hat x_t + a_t,  \end{eqnarray*}

 where :math:`E a_t a_t^\prime \equiv \Omega =  G \Sigma G^\prime + R`.

Factorization of Likelihood Function
====================================

Sample of observations :math:`\{y_s\}_{s=0}^T` on a
:math:`(n_y \times 1)` vector.

.. math::

   \begin{eqnarray*}
    f(y_T, y_{T-1}, \ldots, y_0 )&  = &
         f_T(y_T \vert y_{T-1}, \ldots, y_0) f_{T-1}(y_{T-1} \vert
         y_{T-2}, \ldots, y_0) \cdots \cr
        & &  f_1(y_1 \vert y_0)
    f_0(y_0 )  \cr
    & = & g_T(a_T) g_{T-1} (a_{T-1}) \ldots g_1(a_1) f_0(y_0).\end{eqnarray*}

Gaussian Log-Likelihood:

.. math::

   -.5 \sum_{t=0}^T \biggl\{ n_y \ln (2 \pi ) + \ln \vert \Omega _t \vert
         + a_t' \Omega_t^{-1} a_t \biggr\} .

Covariance Generating Functions
===============================

Autocovariance: :math:`C_x(\tau) = E x_t x_{t-\tau}'`

Generating Function:
:math:`S_x(z) = \sum_{\tau = -\infty}^\infty C_x(\tau) z^\tau, z \in C`

Spectral Factorization Identity
===============================

Original state-space representation has too many shocks and implies:

.. math::

   S_y(z) = G (zI - A^o)^{-1} C C^\prime (z^{-1} I - (A^o)^\prime)^{-1}
   G^\prime + R .

Innovations representation has as many shocks as dimension of
:math:`y_t` and implies

.. math::

   S_y(z) = [G(zI-A^o)^{-1}K +I] [G \Sigma G^\prime + R]
   [K^\prime (z^{-1} I -A^{o\prime})^{-1} G^\prime + I].

.. _spectral-factorization-identity-1:

Spectral Factorization Identity
===============================

Equating these two leads to:

.. math::

   \begin{eqnarray*}
    & & G (zI -  A^o)^{-1} C C^\prime (z^{-1} I - A^{o\prime})^{-1} G^\prime + R = \cr
   &  &[G(zI-A^o)^{-1}K +I] [G \Sigma G^\prime + R] [K'(z^{-1} I -A^{o\prime})^{-1}
   G^\prime + I] .\end{eqnarray*}

 Key Insight: The zeros of the polynomial
:math:`\det [G(zI-A^o)^{-1}K +I]` all lie inside the unit circle, which
means that :math:`a_t` lies in the space spanned by square summable
linear combinations of :math:`y^t`.

.. math:: H(a^t) = H(y^t)

Key Property: Invertibility

Lag Operator
============

.. math:: L x_t \equiv x_{t-1}

.. math:: L^{-1} x_t \equiv x_{t+1}

Wold and Vector Autoregressive Representations
==============================================

A Wold moving average representation for :math:`\{y_t\}` is

.. math:: y_t = [ G(I-A^oL)^{-1}KL + I] a_t  .

 Applying the inverse of the operator on the right side and using

.. math:: [G(I-A^oL)^{-1}KL+I]^{-1} = I - G[I - (A^o-KG)L]^{-1}KL

 gives the vector autoregressive representation

.. math:: y_t = \sum_{j=1}^\infty G (A^o -KG)^{j-1} K y_{t-j} + a_t  .

.. _wold-and-vector-autoregressive-representations-1:

Wold and Vector Autoregressive Representations
==============================================

A Wold moving average representation for :math:`\{y_t\}` is

.. math:: y_t = [ G(I-A^oL)^{-1}{\color{red}K}L + I] a_t  .

 Applying the inverse of the operator on the right side and using

.. math:: [G(I-A^oL)^{-1}{\color{red}K}L+I]^{-1} = I - G[I - (A^o-{\color{red}K}G)L]^{-1}{\color{red}K} L

 gives the vector autoregressive representation

.. math:: y_t = \sum_{j=1}^\infty G (A^o - {\color{red}K}G)^{j-1} {\color{red}K} y_{t-j} + a_t  .

Dynamic Demand Curves and Canonical Household Technologies
===========================================================

Canonical Household Technologies
================================

.. math::

   \begin{eqnarray*}
    h_t &=  &\Delta_h h_{t-1} + \Theta_h  c_t \cr
                s_t & = & \Lambda h_{t-1} + \Pi c_t  \cr
                b_t  &= &U_b z_t. \end{eqnarray*}

Definition: A household service technology
:math:`(\Delta_h, \Theta_h, \Pi,\Lambda, U_b)` is said to be canonical
if


:math:`\Pi` is nonsingular, and

The absolute values of the eigenvalues of :math:`(\Delta_h - \Theta_h
\Pi^{-1}\Lambda)` are strictly less than :math:`1/\sqrt\beta`.

.. _canonical-household-technologies-1:

Canonical Household Technologies
================================

.. math::

   \begin{eqnarray*}
    h_t &=  &\Delta_h h_{t-1} + \Theta_h  c_t \cr
                s_t & = & \Lambda h_{t-1} + \Pi c_t  \cr
                b_t  &= &U_b z_t. \end{eqnarray*}

Definition: A household service technology
:math:`(\Delta_h, \Theta_h, \Pi,\Lambda, U_b)` is said to be canonical
if :math:`\color{red}{\Pi}` is nonsingular, and

The absolute values of the eigenvalues of
:math:`\color{red}{(\Delta_h - \Theta_h
\Pi^{-1}\Lambda)}` are strictly less than :math:`1/\sqrt\beta`.

.. _canonical-household-technologies-2:

Canonical Household Technologies
================================

| Key property (invertiblility): A canonical household service
  technology maps a service process :math:`\{s_t\}` in :math:`L_0^2`
  into a corresponding consumption process :math:`\{c_t\}` for which the
  implied household capital stock process :math:`\{h_t\}` is also in
  :math:`L^2_0`.

An inverse household technology:

.. math::

   \begin{eqnarray*}
    c_t &= & - \Pi^{-1} \Lambda h_{t-1} + \Pi^{-1} s_t\cr
   h_t &= & (\Delta_h - \Theta_h\Pi^{-1} \Lambda) h_{t-1} + \Theta_h \Pi^{-1}
   s_t . \end{eqnarray*}

 Restriction (ii) on the eigenvalues of the matrix
:math:`(\Delta_h - \Theta_h \Pi^{-1}
\Lambda)` keeps the household capital stock :math:`\{h_t\}` in
:math:`L_0^2`.

.. _canonical-household-technologies-3:

Canonical Household Technologies
================================

| Key property (invertiblility): A canonical household service
  technology maps a service process :math:`\{s_t\}` in :math:`L_0^2`
  into a corresponding consumption process :math:`\{c_t\}` for which the
  implied household capital stock process :math:`\{h_t\}` is also in
  :math:`L^2_0`.

An inverse household technology:

.. math::

   \begin{eqnarray*}
    \{ c_t} &= & - \Pi^{-1} \Lambda h_{t-1} + \Pi^{-1} \{ s_t}\cr
   h_t &= & (\Delta_h - \Theta_h\Pi^{-1} \Lambda) h_{t-1} + \Theta_h \Pi^{-1}
   s_t . \end{eqnarray*}

 Restriction (ii) on the eigenvalues of the matrix
:math:`(\Delta_h - \Theta_h \Pi^{-1}
\Lambda)` keeps the household capital stock :math:`\{h_t\}` in
:math:`L_0^2`.

.. _canonical-household-technologies-4:

Canonical Household Technologies
================================

| Key property (invertiblility): A canonical household service
  technology maps a service process :math:`\{s_t\}` in :math:`L_0^2`
  into a corresponding consumption process :math:`\{c_t\}` for which the
  implied household capital stock process :math:`\{h_t\}` is also in
  :math:`L^2_0`.

An inverse household technology:

.. math::

   \begin{eqnarray*}
    c_t &= & - {\color{red}\Pi^{-1}} \Lambda h_{t-1} + {\color{red}\Pi^{-1}} s_t\cr
   h_t &= &{\color{red}(\Delta_h - \Theta_h\Pi^{-1} \Lambda)} h_{t-1} + \Theta_h {\color{red}\Pi^{-1}}
   s_t . \end{eqnarray*}

 Restriction (ii) on the eigenvalues of the matrix
:math:`(\Delta_h - \Theta_h \Pi^{-1}
\Lambda)` keeps the household capital stock :math:`\{h_t\}` in
:math:`L_0^2`.

Dynamic Demand Functions
========================

.. math::

   \rho^0_t \equiv \Pi^{-1 \prime} \Bigl[p^0_t - \Theta _h^\prime E_t
   \sum^\infty_{\tau=1} \beta^\tau (\Delta_h^\prime - \Lambda^\prime \Pi^{-1 \prime}
   \Theta_h^\prime )^{\tau-1}\Lambda^\prime \Pi^{-1 \prime} p^0_{t+\tau} \Bigr] .

.. math::

   \begin{eqnarray*}
   s_{i,t}& = & \Lambda h_{i,t-1} \cr
   h_{i,t}& = & \Delta _h h_{i,t-1}, \end{eqnarray*}

 where :math:`h_{i,-1} = h_{-1}`.

.. _dynamic-demand-functions-1:

Dynamic Demand Functions
========================

.. math:: W_0 = E_0\sum^\infty_{t=0}\beta ^t(w^0_t\ell _t + \alpha ^0_t\cdot d_t) + v_0\cdot k_{-1} .

.. math::

   \mu^w_0 = {E_0 \sum^\infty_{t=0} \beta^t \rho^0_t\cdot
    (b_t -s_{i,t}) - W_0 \over E_0 \sum^\infty_{t=0}
   \beta^t \rho^0_t \cdot \rho^0_t} .

.. math::

   \begin{eqnarray*}
    c_t & = & -\Pi^{-1} \Lambda h_{t-1} + \Pi ^{-1} b_t
    - \Pi^{-1} \mu_0^w E_t \{ \Pi^{\prime\, -1} - \Pi^{\prime\, -1}\Theta_h ' \cr
   & & \qquad [I - (\Delta_h ' - \Lambda ' \Pi^{\prime \, -1} \Theta_h ')\beta L^{-1}]
      ^{-1} \Lambda ' \Pi^{\prime -1} \beta L^{-1} \}  p_t^0  \cr
       h_t & = & \Delta_h h_{t-1} + \Theta_h c_t . \end{eqnarray*}

.. _dynamic-demand-functions-2:

Dynamic Demand Functions
========================

.. math:: W_0 = E_0\sum^\infty_{t=0}\beta ^t(w^0_t\ell _t + \alpha ^0_t\cdot d_t) + v_0\cdot k_{-1} .

.. math::

   \mu^w_0 = {E_0 \sum^\infty_{t=0} \beta^t \rho^0_t\cdot
    (b_t -s_{i,t}) - W_0 \over E_0 \sum^\infty_{t=0}
   \beta^t \rho^0_t \cdot \rho^0_t} .

.. math::

   \begin{eqnarray*}
    c_t & = & -\Pi^{-1} \Lambda h_{t-1} + \Pi ^{-1} b_t
    - \Pi^{-1} \mu_0^w E_t \{ \Pi^{\prime\, -1} - \Pi^{\prime\, -1}\Theta_h ' \cr
   & & \qquad [I - (\Delta_h ' - \Lambda ' \Pi^{\prime \, -1} \Theta_h ')\beta L^{-1}]
      ^{-1} \Lambda ' \Pi^{\prime -1} \beta L^{-1} \}  p_t^0  \cr
       h_t & = & \Delta_h h_{t-1} + \Theta_h c_t . \end{eqnarray*}

 This system expresses consumption demands at date :math:`t` as
functions of: (i) time-\ :math:`t` conditional expectations of future
scaled Arrow-Debreu prices :math:`\{p_{t+s}^0\}_{s=0}^\infty`; (ii) the
stochastic process for the household’s endowment :math:`\{d_t\}` and
preference shock :math:`\{b_t\}`, as mediated through the multiplier
:math:`\mu_0^w` and wealth :math:`W_0`; and (iii) past values of
consumption, as mediated through the state variable :math:`h_{t-1}`.

.. _dynamic-demand-functions-3:

Dynamic Demand Functions
========================

.. math:: W_0 = E_0\sum^\infty_{t=0}\beta ^t(w^0_t\ell _t + \alpha ^0_t\cdot d_t) + v_0\cdot k_{-1} .

.. math::

   \mu^w_0 = {E_0 \sum^\infty_{t=0} \beta^t \rho^0_t\cdot
    (b_t -s_{i,t}) - W_0 \over E_0 \sum^\infty_{t=0}
   \beta^t \rho^0_t \cdot \rho^0_t} .

.. math::

   \begin{eqnarray*}
    c_t & = & -\Pi^{-1} \Lambda h_{t-1} + \Pi ^{-1} b_t
    - \Pi^{-1} \{\mu_0^w } E_t \{ \Pi^{\prime\, -1} - \Pi^{\prime\, -1}\Theta_h ' \cr
   & & \qquad [I - (\Delta_h ' - \Lambda ' \Pi^{\prime \, -1} \Theta_h ')\beta L^{-1}]
      ^{-1} \Lambda ' \Pi^{\prime -1} \beta L^{-1} \}  p_t^0  \cr
       h_t & = & \Delta_h h_{t-1} + \Theta_h c_t . \end{eqnarray*}

 This system expresses consumption demands at date :math:`t` as
functions of: (i) time-\ :math:`t` conditional expectations of future
scaled Arrow-Debreu prices :math:`\{p_{t+s}^0\}_{s=0}^\infty`; (ii) the
stochastic process for the household’s endowment :math:`\{d_t\}` and
preference shock :math:`\{b_t\}`, as mediated through the multiplier
:math:`\mu_0^w` and wealth :math:`W_0`; and (iii) past values of
consumption, as mediated through the state variable :math:`h_{t-1}`.

Gorman Aggregation and Engel curves
===================================

We shall explore how the dynamic demand schedule for consumption goods
opens up the possibility of satisfying Gorman’s (1953) conditions for
aggregation in a heterogeneous consumer model. The first equation of our
demand system is an Engel curve for consumption that is linear in the
marginal utility :math:`\mu_0^2` of individual wealth with a coefficient
on :math:`\mu_0^w` that depends only on prices. The multiplier
:math:`\mu_0^w` depends on wealth in an affine relationship, so that
consumption is linear in wealth. In a model with multiple consumers who
have the same household technologies
(:math:`\Delta_h, \Theta_h, \Lambda, \Pi`) but possibly different
preference shock processes and initial values of household capital
stocks, the coefficient on the marginal utility of wealth is the same
for all consumers. Gorman showed that when Engel curves satisfy this
property, there exists a unique community or aggregate preference
ordering over aggregate consumption that is independent of the
distribution of wealth.

Re-Opened Markets
=================

.. math::

   \rho^t_{t} \equiv \Pi^{-1 \prime} \Bigl[p^t_{t} - \Theta _h^\prime E_t
   \sum^\infty_{\tau=1} \beta^\tau (\Delta_h^\prime - \Lambda^\prime \Pi^{-1 \prime}
   \Theta_h^\prime )^{\tau-1}\Lambda^\prime \Pi^{-1 \prime} p^t_{t+\tau} \Bigr] .

.. math::

   \begin{eqnarray*}
    s_{i,t}& = & \Lambda h_{i,t-1} \cr
   h_{i,t}& =  & \Delta _h h_{i,t-1},\end{eqnarray*}

 where now :math:`h_{i,t-1} = h_{t-1}`. Define time :math:`t` wealth
:math:`W_t`

.. math:: W_t = E_t\sum^\infty_{j=0}\beta ^j(w^t_{t+j}\ell_{t+j} + \alpha ^t_{t+j}\cdot d_{t+j}) + v_t\cdot k_{t-1} .

.. math::

   \mu^w_t = {E_t \sum^\infty_{j=0} \beta^j \rho^t_{t+j}\cdot
    (b_{t+j} -s_{i,t+j}) - W_t \over E_t \sum^\infty_{t=0}
   \beta^j \rho^t_{t+j} \cdot \rho^t_{t+j}} .

.. math::

   \begin{eqnarray*}
    c_t & = & -\Pi^{-1} \Lambda h_{t-1} + \Pi ^{-1} b_t
    - \Pi^{-1} \mu_t^w E_t \{ \Pi^{\prime\, -1} - \Pi^{\prime\, -1}\Theta_h ' \cr
   & & \qquad [I - (\Delta_h ' - \Lambda ' \Pi^{\prime \, -1} \Theta_h ')\beta L^{-1}]
      ^{-1} \Lambda ' \Pi^{\prime -1} \beta L^{-1} \}  p_t^t  \cr
       h_t & = & \Delta_h h_{t-1} + \Theta_h c_t .  \end{eqnarray*}

Dynamic Demand
==============

Define a time :math:`t` continuation of a sequence
:math:`\{z_t\}_{t=0}^\infty` as the sequence
:math:`\{z_\tau\}_{\tau=t}^\infty`. The demand system indicates that the
time :math:`t` vector of demands for :math:`c_t` are influenced by:

Through the multiplier :math:`\mu^w_t`, the time :math:`t` continuation
of the preference shock process :math:`\{b_t\}` and the time :math:`t`
continuation of :math:`\{s_{i,t}\}`.

The time :math:`t-1` level of household durables :math:`h_{t-1}`.

Everything that affects the household’s time :math:`t` wealth, including
its stock of physical capital :math:`k_{t-1}` and its value :math:`v_t`,
the time :math:`t` continuation of the factor prices
:math:`\{w_t, \alpha_t\}`, the household’s continuation endowment
process, and the household’s continuation plan for :math:`\{\ell_t\}`.

The time :math:`t` continuation of the vector of prices
:math:`\{p_t^t\}`.

Attaining a canonical hh technology
===================================

Apply the following version of a factorization identity:

.. math::

   \begin{eqnarray*}
    [\Pi &+ & \beta^{1/2} L^{-1} \Lambda (I - \beta^{1/2} L^{-1}
   \Delta_h)^{-1} \Theta_h]^\prime [\Pi + \beta^{1/2} L
   \Lambda (I - \beta^{1/2} L \Delta_h)^{-1} \Theta_h]\cr
   &=& [\hat\Pi + \beta^{1/2} L^{-1} \hat\Lambda
   (I - \beta^{1/2} L^{-1} \Delta_h)^{-1} \Theta_h]^\prime
   [\hat\Pi + \beta^{1/2} L \hat\Lambda
   (I - \beta^{1/2} L \Delta_h)^{-1} \Theta_h]\end{eqnarray*}

 The factorization identity guarantees that the
:math:`[\hat \Lambda, \hat \Pi]` representation satisfies both
requirements for a canonical representation.



Examples: Partial Equilibrium
=============================

Demand:

.. math::

   \begin{eqnarray*}
     c_t &  = &  -\Pi^{-1} \Lambda h_{t-1} + \Pi ^{-1} b_t - \Pi^{-1}
       \mu_0^w E_t \{ \Pi^{\prime\, -1} - \Pi^{\prime\, -1}\Theta_h' \cr
     & & \qquad[I - (\Delta_h' - \Lambda' \Pi^{\prime\, -1} \Theta_h')\beta
        L^{-1}]^{-1} \Lambda' \Pi^{\prime -1} \beta L^{-1} \}  p_t  \cr
     h_t & = & \Delta_h h_{t-1} + \Theta_h c_t .  \end{eqnarray*}

.. _examples-partial-equilibrium-1:

Examples: Partial Equilibrium
=============================

A representative firm takes as given and beyond its control the
stochastic process :math:`\{p_t\}_{t=0}^\infty`. The firm sells its
output :math:`c_t` in a competitive market each period. Only spot
markets convene at each date :math:`t\geq 0`. The firm also faces an
exogenous process of cost disturbances :math:`d_t`.

The firm chooses stochastic processes
:math:`\{c_t, g_t, i_t, k_t\}_{t=0}^\infty` to maximize

.. math:: E_0 \sum_{t=0}^\infty \beta^t \{ p_t \cdot c_t - g_t \cdot g_t/2 \}

 subject to given :math:`k_{-1}` and

.. math::

   \begin{eqnarray*}
   \Phi_c c_t  +  \Phi_i i_t + \Phi_g g_t & = &\Gamma k_{t-1} + d_t  \cr
      k_t& =&  \Delta_k k_{t-1} + \Theta_k i_t . \cr
    %  x_{t+1}& = A^o x_t + C w_{t+1}  \cr
    %  d_t& = S_d x_t  \cr
    %  p_t& = M_c x_t \cr
                     \end{eqnarray*}

Equilibrium Investment Under Uncertainty
========================================

A representative firm maximizes

.. math:: E \sum_{t=0}^\infty \beta^t \{ p_t c_t - g_t^2/2 \} ,

 subject to the technology

.. math::

   \begin{eqnarray*}
    c_t &= & \gamma k_{t-1} \cr
                k_t &=  & \delta_k k_{t-1} + i_t \cr
                g_t & = & f_1 i_t + f_2 d_t , \end{eqnarray*}

 where :math:`d_t` is a cost shifter, :math:`\gamma> 0`, and
:math:`f_1 >0` is a cost parameter and :math:`f_2 =1`. Demand is
governed by

.. math:: p_t = \alpha_0 - \alpha_1 c_t + u_t,

 where :math:`u_t` is a demand shifter with mean zero and
:math:`\alpha_0, \alpha_1` are positive parameters. Assume that
:math:`u_t, d_t` are uncorrelated first-order autoregressive processes.

A Rosen-Topel Housing Model
===========================

.. math::

   \begin{eqnarray*}
    R_t &= & b_t + \alpha h_t \cr
                p_t & = & E_t \sum_{\tau =0}^\infty (\beta \delta_h)^\tau
                          R_{t+\tau} \end{eqnarray*}

 where :math:`h_t` is the stock of housing at time :math:`t`,
:math:`R_t` is the rental rate for housing, :math:`p_t` is the price of
new houses, and :math:`b_t` is a demand shifter; :math:`\alpha < 0` is a
demand parameter, and :math:`\delta_h` is a depreciation factor for
houses.

.. _a-rosen-topel-housing-model-1:

A Rosen-Topel Housing Model
===========================

We cast this demand specification within our class of models by letting
the stock of houses :math:`h_t` evolve according to

.. math:: h_t = \delta_h h_{t-1} + c_t, \quad \delta_h \in (0,1) ,

 where :math:`c_t` is the rate of production of new houses. Houses
produce services :math:`s_t` according to
:math:`s_t  = \bar \lambda h_t` or
:math:`s_t  = \lambda h_{t-1} + \pi c_t,` where
:math:`\lambda= \bar \lambda \delta_h, \pi = \bar \lambda`. We can take
:math:`\bar \lambda \rho_t^0  = R_t` as the rental rate on housing at
time :math:`t`, measured in units of time :math:`t` consumption
(housing).

.. _a-rosen-topel-housing-model-2:

A Rosen-Topel Housing Model
===========================

Demand for housing services is

.. math:: s_t = b_t - \mu_0 \rho_t^0 ,

 where the price of new houses :math:`p_t` is related to
:math:`\rho_t^0` by
:math:`\rho_t^0 = \pi^{-1} [  p_t - \beta \delta_h E_t p_{t+1}] .`

Cattle Cycles
=============

Rosen, Murphy, and Scheinkman (1994). Let :math:`p_t` be the price of
freshly slaughtered beef, :math:`m_t` the feeding cost of preparing an
animal for slaughter, :math:`\tilde h_t` the one-period holding cost for
a mature animal, :math:`\gamma_1 \tilde h_t` the one-period holding cost
for a yearling, and :math:`\gamma_0 \tilde
h_t` the one-period holding cost for a calf. The cost processes
:math:`\{\tilde h_t, m_t\}_{t=0}^\infty` are exogenous, while the
stochastic process :math:`\{p_t\}_{t=0}^\infty` is determined by a
rational expectations equilibrium. Let :math:`\tilde x_t` be the
breeding stock, and :math:`\tilde y_t` be the total stock of animals.
The law of motion for cattle stocks is

.. math:: \tilde x_t = (1-\delta) \tilde x_{t-1} + g \tilde x_{t-3} - c_t,

 where :math:`c_t` is a rate of slaughtering. The total head count of
cattle,

.. math:: \tilde y_t = \tilde x_t + g \tilde x_{t-1} + g \tilde x_{t-2},

 is the sum of adults, calves, and yearlings, respectively.

.. _cattle-cycles-1:

Cattle Cycles
=============

A representative farmer chooses :math:`\{c_t, \tilde x_t\}` to maximize

.. math::

   \begin{eqnarray*}
    E_0 \sum_{t=0}^\infty \beta^t \{ p_t c_t & - &
        \tilde h_t \tilde x_t
           -(\gamma_0 \tilde h_t) (g \tilde x_{t-1}) - (\gamma_1 \tilde h_t)
            (g \tilde x_{t-2}) - m_t c_t \cr
            &  - &   \Psi(\tilde x_t, \tilde x_{t-1},
            \tilde x_{t-2}, c_t) \}, \end{eqnarray*}

 where

.. math::

   \Psi = {\psi_1 \over 2} \tilde x_t^2 + {\psi_2 \over 2} \tilde x_{t-1}^2
         + {\psi_3 \over 2} \tilde x_{t-2}^2 + {\psi_4 \over 2} c_t^2.

Demand is governed by

.. math:: c_t = \alpha_0 - \alpha_1 p_t + \tilde d_t ,

 where :math:`\alpha_0 > 0`, :math:`\alpha_1 > 0`, and
:math:`\{\tilde d_t\}_{t=0}^\infty` is a stochastic process with mean
zero representing a demand shifter.

Models of Occupational Choice and Pay
=====================================

-  Rosen schooling model for engineers.

-  Two-occupation model.

Market for Engineers
====================

Ryoo and Rosen’s (2004) model consists of the following equations:
first, a demand curve for engineers

.. math:: w_t = - \alpha_d N_t + \epsilon_{1t}\ ,\ \alpha_d > 0 ;

 second, a time-to-build structure of the education process

.. math:: N_{t+k} = \delta_N N_{t+k-1} + n_t\ ,\ 0<\delta_N<1;

 third, a definition of the discounted present value of each new
engineering student

.. math::

   v_t = \beta^k E_t \sum^\infty_{j=0} (\beta  \delta_N)^j
      w_{t+k+j};

 and fourth, a supply curve of new students driven by :math:`v_t`

.. math:: n_t = \alpha_s v_t + \epsilon_{2t}\ ,\ \alpha_s > 0.

 Here :math:`\{\epsilon_{1t}, \epsilon_{2t}\}` are stochastic processes
of labor demand and supply shocks.

.. _market-for-engineers-1:

Market for Engineers 
=====================

Definition: A partial equilibrium is a stochastic process
:math:`\{w_t, N_t, v_t, n_t\}^\infty_{t=0}` satisfying these four
equations, and initial conditions
:math:`N_{-1}, n_{-s}, s=1, \ldots, -k`.

Capturing the Market for Engineers
==================================

We sweep the time-to-build structure and the demand for engineers into
the household technology and putting the supply of new engineers into
the technology for producing goods.

.. math::

   \begin{eqnarray*}
    s_t &= & [\lambda_1 \ 0 \ \ldots \ 0]\ \begin{bmatrix}
   h_{1t-1}\cr h_{2t-1}\cr \vdots \cr h_{k+1,t-1}\end{bmatrix} + 0 \cdot c_t \cr
   \begin{bmatrix} h_{1t}\cr h_{2t}\cr \vdots\cr h_{k,t} \cr
      h_{k+1,t}\end{bmatrix} &= &
   \begin{bmatrix} \delta_N & 1 & 0 & \cdots & 0\cr 0 & 0 & 1 & \cdots & 0\cr
   \vdots & \vdots & \vdots & \ddots & \vdots\cr 0 & \cdots & \cdots & 0 & 1\cr
   0 & 0 & 0 & \cdots & 0 \end{bmatrix} \begin{bmatrix}h_{1t-1}\cr h_{2t-1}\cr \vdots\cr h_{k,t-1} \cr
         h_{k+1,t-1}\end{bmatrix} + \begin{bmatrix}0\cr 0\cr \vdots\cr  0\cr 1\cr\end{bmatrix}  c_t \cr
   %b_t &=  & \epsilon_{1t}
    \end{eqnarray*}

 This specification sets Rosen’s :math:`N_t = h_{1t-1}, n_t = c_t,
h_{\tau+1,t-1} = n_{t-\tau}, \tau=1, \ldots, k`, and uses the
home-produced service to capture the demand for labor. Here
:math:`\lambda_1` embodies Rosen’s demand parameter :math:`\alpha_d`.

Trick for Capturing the Market for Engineers
============================================

The supply of new workers becomes our consumption. The dynamic demand
curve becomes Rosen’s dynamic supply curve for new workers.


Remark: This has an Imai-Keane flavor.

Skilled and Unskilled Workers
=============================

First, a demand curve for labor

.. math::

   \begin{bmatrix} w_{ut} \cr w_{st} \end{bmatrix}
        = \alpha_d \begin{bmatrix} N_{ut} \cr N_{st} \end{bmatrix}
           + \epsilon_{1t} ;

 where :math:`\alpha_d` is a :math:`(2 \times 2)` matrix of demand
parameters and :math:`\epsilon_{1t}` is a vector of demand shifters;
second, time-to-train specifications for skilled and unskilled labor,
respectively:

.. math::

   \begin{eqnarray*}
    N_{st+k} &= &\delta_N N_{st+k-1} + n_{st} \cr
                N_{ut} &=  &\delta_N N_{ut-1} + n_{ut} ; \end{eqnarray*}

 where :math:`N_{st}, N_{ut}` are stocks of the two types of labor, and
:math:`n_{st}, n_{ut}` are entry rates into the two occupations;

.. _skilled-and-unskilled-workers-1:

Skilled and Unskilled Workers
=============================

third, definitions of discounted present values of new entrants to the
skilled and unskilled occupations, respectively:

.. math::

   \begin{eqnarray*}
    v_{st} &= & E_t \beta^k \sum_{j=0}^\infty (\beta \delta_N)^j
            w_{st+k+j} \cr
                v_{ut} & =  &E_t \sum_{j=0}^\infty (\beta \delta_N)^j
       w_{ut+j}, \end{eqnarray*}

 where :math:`w_{ut}, w_{st}` are wage rates for the two occupations;
and fourth, supply curves for new entrants:

.. math::

   \begin{bmatrix}n_{st} \cr n_{ut}\end{bmatrix}
         = \alpha_s \begin{bmatrix} v_{ut} \cr v_{st} \end{bmatrix} +
           \epsilon_{2t}.

Short Cut
=========

As an alternative, Siow simply used the ‘equalizing differences’
condition

.. math:: v_{ut} = v_{st}.


Permanent Income Models
=======================

-  Many consumption goods and services.

-  A single capital good with ‘:math:`R \beta =1`’.

.. math::

   \begin{eqnarray*}
    \phi_c \cdot c_t+i_t& = &\gamma k_{t-1}+e_t \cr
               k_t& = & k_{t-1} + i_t \end{eqnarray*}

.. math:: \phi_ii_t-g_t=0

Permanent Income Models: Implication One
========================================

Equality of Present Values of Moving Average Coefficients of :math:`c`
and :math:`e`

.. math:: k_{t-1} = \beta \sum\limits_{j=0}^\infty \beta^j (\phi_c \cdot c_{t+j} - e_{t+j}).

 :math:`\Rightarrow`

.. math::

   k_{t-1} = \beta \sum\limits_{j=0}^\infty  \beta^j E (\phi_c
       \cdot c_{t+j} - e_{t+j})|J_t.

 :math:`\Rightarrow`

.. math::

   \sum\limits_{j=0}^\infty \beta^j (\phi_c)^\prime \chi_j =
    \sum\limits_{j=0}^\infty \beta^j \epsilon_j

 where :math:`\chi_j w_t` is the response of :math:`c_{t+j}` to
:math:`w_t` and :math:`\epsilon_j w_t` is the response of endowment
:math:`e_{t+j}` to :math:`w_t`:

Permanent Income Models: Implication Two
========================================

Martingales

.. math::

   \begin{eqnarray*}
    {\mathcal M}_t^k  & = & E ({\mathcal M}_{t+1}^k | J_t) \cr
   {\mathcal M}_t^e  & = & E ({\mathcal M}_{t+1}^e | J_t) \end{eqnarray*}

and

.. math:: {\mathcal M}_t^c  =  (\Phi_c)^\prime {\mathcal M}_t^d = \phi_c {\cal M}_t^e

Permanent Income Models: Testing
================================

Test the two implications:

-  Equality of present values of moving average coefficients.

-  Martingale :math:`{\mathcal M}_t^k`.

These have been tested in work by Hansen, Sargent, and Roberds (1991)
and by Attanasio and Pavoni (2011).



Gorman Heterogeneous Households
===============================

.. math::

   -\ \left({1 \over 2}\right)\ E \sum_{t=0}^\infty\, \beta^t\, \bigl[(s_{jt} -
   b_{jt}) \cdot (s_{jt} - b_{jt}) + \ell_{jt}^{2}\bigr] \mid J_0

.. math:: s_{jt} = \Lambda\, h_{j,t-1} + \Pi\, c_{jt}

.. math:: h_{jt} = \Delta_h\, h_{j,t-1} + \Theta_h\, c_{jt}

 and :math:`h_{j,-1}` is given.

.. math:: b_{jt} = U_{bj} z_t ,

.. math::

   E\, \sum_{t=0}^\infty\, \beta^t\, p_t^0\, \cdot c_{jt} \mid J_0 = E\,
   \sum_{t=0}^\infty\, \beta^t\, (w_t^0\, \ell_{jt} + \alpha_t^0\, \cdot
        d_{jt}) \mid
   J_0 + v_0\, \cdot k_{j,-1},

 where :math:`k_{j,-1}` is given. The :math:`j^{\rm th}` consumer owns
an endowment process :math:`d_{jt}`, governed by the stochastic process
:math:`d_{jt} = U_{dj}\, z_t .`

.. _gorman-heterogeneous-households-1:

Gorman Heterogeneous Households
===============================

This specification confines heterogeneity among consumers to: (a)
differences in the preference processes :math:`\{b_{jt}\}`, represented
by different selections of :math:`U_{bj}`; (b) differences in the
endowment processes :math:`\{d_{jt}\}`, represented by different
selections of :math:`U_{dj}`; (c) differences in :math:`h_{j,-1}`; and
(d) differences in :math:`k_{j,-1}`. The matrices
:math:`\Lambda,\,\Pi,\,\Delta_h,\,\Theta_h` do not depend on :math:`j`.
This makes everybody’s demand system have the form described earlier,
with different :math:`\mu_{j0}^w`\ ’s (reflecting different wealth
levels) and different :math:`b_{jt}` preference shock processes and
initial conditions for household capital stocks.   


Punchline: :math:`\exists` a representative consumer. Use it to compute
competitive equilibrium aggregate allocation and price system.

To Compute Individual Allocations
=================================

Set

.. math:: \ell_{jt} = (\mu_{0j}^w/\mu_{0a}^w) \ell_{at}

 Then solve the following equation for :math:`\mu_{0j}^{w}`:

.. math::

   \mu_{0j}^{w}  E_0 \sum_{t=0}^\infty \beta^t \{\rho_t^0 \cdot \rho_t^0
       + (w_t^0/ \mu_{0a}^{w}) \ell_{at} \}
       = E_0 \sum_{t=0}^\infty \beta^t \{ \rho_t^0 \cdot (b_{jt} - s_{jt}^i)
       -  \alpha_t^0 \cdot d_{jt} \}
          - v_0 k_{j,-1}

.. math:: s_{jt} - b_{jt} = \mu_{0j}^w\rho^0_t

.. math::

   \begin{eqnarray*}
    c_{jt} &= & - \Pi^{-1} \Lambda h_{j,t-1} + \Pi^{-1}s_{jt} \cr
   h_{jt} &= & (\Delta_h - \Theta_h \Pi^{-1}\Lambda) h_{j,t-1} + \Pi^{-1}
       \Theta_h  s_{jt} \end{eqnarray*}

 :math:`h_{j,-1}` given.


Non-Gorman Heterogeneous Households
===================================

Preferences and Household Technologies:

.. math::

   - {1\over 2} E\, \sum^\infty_{t=0}\, \beta^t\, [ (s_{it} - b_{it}) \cdot
   (s_{it} - b_{it}) + \ell^2_{it}]\mid J_0 .

.. math::

   \begin{eqnarray*}
    s_{it} &= & \Lambda_i h_{i t-1} + \Pi_i\, c_{it} \cr
   h_{it} &=  &\Delta_{h_i}\, h_{i t-1} + \Theta_{h_i} c_{it}\ ,\ i=1,2 .\end{eqnarray*}

.. math:: b_{it} = U_{bi} z_t

.. math:: z_{t+1} = A_{22} z_t + C_2 w_{t+1} .

Production Technology
=====================

.. math::

   \Phi_c (c_{1t} + c_{2t}) + \Phi_g g_t + \Phi_i i_t = \Gamma
   k_{t-1} + d_{1t} + d_{2t}

.. math:: k_t = \Delta_k k_{t-1} + \Theta_k i_t

.. math::

   g_t \cdot g_t = \ell^2_t,\qquad
      \ \ell_t = \ell_{1t} + \ell_{2t}  .

.. math:: d_{it} = U_{d_i} z_t\quad ,\ i=1,2 .

A Pareto Problem
================

.. math::

   \begin{eqnarray*}
    & & - {1\over 2}\, \lambda E_0 \sum^\infty_{t=0}\, \beta^t [ (s_{1t}
   - b_{1t})\cdot (s_{1t} - b_{1t}) + \ell^2_{1t}]\cr
    & &
   - {1\over 2}\, (1-\lambda) E_0 \sum^\infty_{t=0}\, \beta^t [ (s_{2t} -
   b_{2t}) \cdot (s_{2t} - b_{2t}) + \ell^2_{2t}] \end{eqnarray*}

Mongrel Aggregation: Static
===========================

Single consumer static inverse demand and implied preferences:

.. math:: c_t = \Pi^{-1} b_t - \mu_0 \Pi^{-1} \Pi^{-1 \prime} p_t

 An inverse demand curve is

.. math:: p_t = \mu_0^{-1} \Pi' b_t - \mu_0^{-1} \Pi' \Pi c_t.

 Integrating the marginal utility vector shows that preferences can be
taken to be

.. math:: ( - 2 \mu_0)^{-1} (\Pi c_t - b_t) \cdot (\Pi c_t - b_t )

.. _mongrel-aggregation-static-1:

Mongrel Aggregation: Static
===========================

Key Insight: Factor the inverse of a ‘covariance matrix’.

Two consumers, :math:`i=1,2`, with demand curves

.. math:: c_{it} = \Pi_i^{-1} b_{it} - \mu_{0i} \Pi_i^{-1} \Pi_i^{-1 \prime} p_t.

.. math::

   c_{1t} + c_{2t} = (\Pi_1^{-1} b_{1t} + \Pi_2^{-1} b_{2t})
       - (\mu_{01} \Pi_1^{-1} \Pi_1^{-1 \prime} + \mu_{02} \Pi_2
          \Pi_2^{-1 \prime}) p_t .

 Setting :math:`c_{1t} + c_{2t} = c_t` and solving for :math:`p_t` gives

.. math::

   \begin{eqnarray*}
    p_t &= & (\mu_{01} \Pi_1^{-1} \Pi_1^{-1 \prime} + \mu_{02}
       \Pi_2^{-1} \Pi_2^{-1 \prime})^{-1}
         (\Pi_1^{-1} b_{1t} + \Pi_2^{-1} b_{2t}) \cr
     &- & (\mu_{01} \Pi_1^{-1} \Pi_1^{-1 \prime} +
        \mu_{02} \Pi_2^{-1} \Pi_2^{-1 \prime}
         )^{-1} c_t. \end{eqnarray*}

Punchline: choose :math:`\Pi` associated with the aggregate ordering to
satisfy

.. math::

   \mu_0^{-1} \Pi' \Pi = (\mu_{01} \Pi_1^{-1} \Pi_2^{-1 \prime}
         + \mu_{02} \Pi_2^{-1} \Pi_2^{-1 \prime})^{-1} .

Dynamic Analogue
================

-  Static: factor a covariance matrix like object.

-  Dynamic: factor a spectral-density matrix like object.

Programming Problem for Dynamic Mongrel Aggregation
===================================================

Our strategy for deducing the mongrel preference ordering over
:math:`c_t = c_{1t} + c_{2t}` is to solve the programming problem:
choose :math:`\{c_{1t},c_{2t}\}` to maximize the criterion

.. math::

   \sum^\infty_{t=0} \beta^t [\lambda (s_{1t} - b_{1t}) \cdot (s_{1t} - b_{1t})
    + (1-\lambda) (s_{2t} - b_{2t}) \cdot (s_{2t} - b_{2t})]

 subject to

.. math::

   \begin{eqnarray*}
    h_{jt} &= & \Delta_{hj}\, h_{jt-1} + \Theta_{hj}\, c_{jt}, j=1,2\cr
   s_{jt} &=  &\Delta_j h_{jt-1} + \Pi_j c_{jt}\ , j=1,2\cr
   c_{1t} +   c_{2t} & = &c_t, \end{eqnarray*}

 subject to :math:`(h_{1, -1},\, h_{2, -1})` given and
:math:`\{b_{1t}\},\, \{b_{2t}\},\, \{c_t\}` being known and fixed
sequences. Substituting the :math:`\{c_{1t},\, c_{2t}\}` sequences that
solve this problem as functions of :math:`\{b_{1t},\, b_{2t},\, c_t\}`
into the objective determines a mongrel preference ordering over
:math:`\{c_t\} = \{c_{1t} + c_{2t}\}`.

Dynamic Case: A Programming Problem for Mongrel Aggregation
===========================================================

In solving this problem, it is convenient to proceed by using Fourier
transforms.   

Secret Weapon: Another application of the spectral factorization
identity.

.. _section-4:

   Complete market economies are all alike but each incomplete market
   economy is incomplete in its own individual way.   Robert E. Lucas,
   Jr., (1989)

.. _section-5:

 


.. _section-6:

 


.. _section-7:

 

.. _section-8:

 
