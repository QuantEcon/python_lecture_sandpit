.. _geom_series:

.. include:: /_static/includes/lecture_howto_py.raw

.. index::
    single: python
    
**********************
Geometric series 101
**********************

This notebook describes important sets of ideas in economics that rely
on using the mathematics of geometric series

Among these are

-  the Keynesian multiplier

-  the money multiplier that prevails in fractional reserve banking
   systems

-  interest rates and present values of streams of payouts from assets

These and other applications illustrate the wise crack that **in
economics, a little knowledge of geometric series goes a long way**

Geometric series: key formulas
===============================

To start, we let :math:`c` be a real number that lies strictly between
:math:`-1` and :math:`1`

-  We often write this as :math:`c \in (-1,1)`

-  here :math:`(-1,1)` denotes the collection of all real numbers that
   are strictly less than :math:`1` and strictly greater
   than\ :math:`-1`

-  the symbol :math:`\in` means *in* or *belongs to the following
   set*

We want to evaluate geometric series of two types -- infinite and finite

Infinite geometric series
--------------------------

The first type of geometric that interests us is the infinite series

.. math:: 1 + c + c^2 + c^3 + \cdots

Where :math:`\cdots` means that the series contiues without limit.

The key formula is

.. math::
  :label: infinite
  
  1 + c + c^2 + c^3 + \cdots = \frac{1}{1 -c }

**How to prove key formula :eq:`infinite`:**

Multiply both sides of the above equation by :math:`(1-c)` and verify
that if :math:`c \in (-1,1)`, then the outcome is the
equation :math:`1 = 1`

Finite geometric series
------------------------

The second series that interests us is the finite geomtric series

.. math:: 1 + c + c^2 + c^3 + \cdots + c^T 

where :math:`T` is a positive integer

The key formula here is

.. math:: 1 + c + c^2 + c^3 + \cdots + c^T  = \frac{1 - c^{T+1}}{1-c}

**Remark:** The above formula works for any value of the scalar
:math:`c`. We don't have to restrict :math:`c` to be in the
set :math:`(-1,1)`

Three Examples
===============

We now move on to describe some famuous economic applications of
geometric series

The money multiplier in fractional reserve banking
--------------------------------------------------

In a fractional reserve banking system, banks hold only a fraction
:math:`r \in (0,1)` of cash behind each **deposit receipt** that they
issue

-  in recent times

   -  cash consists of pieces of paper issued by the government and
      called dollars or pounds or :math:`\ldots`

   -  a *deposit* is a balance in a checking or savings account that
      entitles the owner to ask the bank for immediate payment in cashs

-  when the UK and France and the US were on either a gold or silver
   standard (before 1914, for example)

   -  cash was a gold or silver coin

   -  a *deposit receipt* was a *bank note* that the bank promised to
      convert into gold or silver on demand; (sometimes it was also a
      checking or savings account balance)

Economists and financiers often define the **supply of money** as an
economy-wide sum of **cash** plus **deposits**

In a **fractional reserve banking system** (one in which the reserve
ratio :math:`r < 1`), **banks create money** by issuing deposits
*backed* by fractional reserves plus loans that they make to their
customers

A geometric series is a key tool for understanding how banks create
money (i.e., deposits) in a fractional reserve system

The geometric series formula :eq:`infinite` is at the heart of the classic model of
the money creation process -- one that leads us to the celebrated
**money multiplier**

A simple model
~~~~~~~~~~~~~~

There is a set of banks named :math:`i = 0, 1, 2, \ldots`

Bank :math:`i`'s loans :math:`L_i`, deposits :math:`D_i`, and
reserves :math:`R_i` must satisfy the balance sheet equation (because
**balance sheets balance**):

.. math:: L_i + R_i = D_i

The left side of the above equation is the sum of the bank's **assets**,
namely, the loans :math:`L_i` it has outstanding plus its reserves of
cash :math:`R_i`. The right side records bank :math:`i`'s liabilities,
namely, the deposits :math:`D_i` held by its depositors; these are
IOU's from the bank to its depositors in the form of either checking
accounts or savings accounts (or before 1914, bank notes issued by a
bank stating promises to redeem note for gold or silver on demand)

.. TO REMOVE:
.. Dongchen: is there a way to add a little balance sheet here? 
.. with assets on the left side and liabilities on the right side?

Ecah bank :math:`i` sets its reserves to satisfy the equation

.. math::
  :label: reserves
  
  R_i = r D_i

where :math:`r \in (0,1)` is its **reserve-deposit ratio** or **reserve
ratio** for short

-  the reserve ratio is either set by a government or chosen by banks
   for precautionary reasons

Next we add a theory stating that bank :math:`i+1`'s deposits depend
entirely on loans made by bank :math:`i`, namely

.. math:: 
  :label: deposits
  
  D_{i+1} = L_i

Thus, we can think of the banks as being arranged along a line with
loans from bank :math:`i` being immediately deposited in :math:`i+1`

-  in this way, the debtors to bank :math:`i` become creditors of
   bank :math:`i+1`

Finally, we add an *initial condition* about an exogenous level of bank
:math:`0`'s deposits

.. math:: D_0 \ \text{ is given exogenously}

We can think of :math:`D_0` as being the amount of cash that a first
depositor put into the first bank in the system, bank number :math:`i=0`

Now we do a little algebra

Combining equations :eq:`reserves` and :eq:`deposits` tells us that

.. math:: 
  :label: fraction
  
  L_i = (1-r) D_i

This states that bank :math:`i` loans a fraction :math:`(1-r)` of its
deposits and keeps a fraction :math:`r` as cash reserves

Combining equation :eq:`fraction` with equation :eq:`deposits` tells us that

.. math:: D_{i+1} = (1-r) D_i  \ \text{ for } i \geq 0

which implies that

.. math::
  :label: geomseries
  
  D_i = (1 - r)^i D_0  \ \text{ for } i \geq 0

Equation :eq:`geomseries` expresses :math:`D_i` as the :math:`i` th term in the
product of :math:`D_0` and the geometric series

.. math::  1, (1-r), (1-r)^2, \cdots

Therefore, the sum of all deposits in our banking system
:math:`i=0, 1, 2, \ldots` is

.. math::
  :label: sumdeposits
  
  \sum_{i=0}^\infty (1-r)^i D_0 =  \frac{D_0}{1 - (1-r)} = \frac{D_0}{r}

**Money multiplier**

The **money multiplier** is a number that tells the multiplicative
factor by which an exogenous injection of cash into bank :math:`0` leads
to an increase in the total deposits in the banking system

Equation :eq:`sumdeposits` asserts that the **money multiplier** is
:math:`\frac{1}{r}`

-  an initial deposit of cash of :math:`D_0` in bank :math:`0` leads
   the banking system to create total deposits of :math:`\frac{D_0}{r}`

-  The initial deposit :math:`D_0` is held as reserves, distributed
   throughout the banking system according to :math:`D_0 = \sum_{i=0}^\infty R_i`

.. Dongchen: can you think of some simple Python examples that 
.. illustrate how to create sequences and so on? Also, some simple 
.. experiments like lowering reserve requirements? Or others you may suggest?


Keynesian multiplier
--------------------

Static version
----------------

The famous economist John Maynard Keynes and his followers created a
simple model intended to determine national income :math:`y` in
circumstances in which

-  there are substantial unemployed resources, in particular **excess
   supply** of labor and capital

-  prices and interest rates fail to adjust to make aggregate **supply
   equal demand** (e.g., prices and interest rates are frozen)

-  national income is entirely determined by aggregate demand

An elementary Keynesian model of national income determination consists
of three equations that describe aggegate demand for :math:`y` and its
components

The first equation is a national income identity asserting that
consumption :math:`c` plus investment :math:`i` equals national income
:math:`y`:

.. math:: c+ i = y

The second equation is a Keynesian consumption function asserting that
people consume a fraction :math:`b \in (0,1)` of their income:

.. math:: c = b y

The fraction :math:`b \in (0,1)` is called the **marginal propensity to
consume**

The fraction :math:`1-b \in (0,1)` is called the **marginal propensity
to save**

The third equation simply states that investment is exogenous at level
:math:`i`

- *exogenous* means *determined outside this model*

Substituting the second equation into the first gives :math:`(1-b) y = i;
 solving this equation for :math:`y` gives

.. math:: y = \frac{1}{1-b} i  

The quantity :math:`\frac{1}{1-b}` is called the **investment
multiplier** or simply the **multiplier**

Applying the formula for the sum of an infinite geometric series, we can
write the above equation as

.. math:: y = i \sum_{t=0}^\infty b^t 

where :math:`t` is a nonnegative integer

So we arrive at the following equivalent expressions for the multiplier:

.. math:: \frac{1}{1-b} =   \sum_{t=0}^\infty b^t 

The expression :math:`\sum_{t=0}^\infty b^t` motivates an interpretation
of the multiplier as the outcome of a dynamic process that we describe
next

Dynamic version of Keynesian multiplier
---------------------------------------

We arrive at a dynamic version by interpreting the nonnegative integer
:math:`t` as indexing time and changing our specification of the
consumption function to take time into account

-  we add a one-period lag in how income affects consumption

We let :math:`c_t` be consumption at time :math:`t` and :math:`i_t` be
investment at time :math:`t`

We modify our consumption function to assume the form

.. math:: c_t = b y_{t-1} 

so that :math:`b` is the marginal propensity to consume (now) out of
last period's income

We begin wtih an initial condition stating that

.. math:: y_{-1} = 0

We also assume that

.. math:: i_t = i \ \ \textrm {for all }  t \geq 0

so that investment is constant over time

It follows that

.. math:: y_0 = i + c_0 = i + b y_{-1} =  i

and

.. math:: y_1 = c_1 + i = b y_0 + i = (1 + b) i 

and

.. math:: y_2 = c_2 + i = b y_1 + i = (1 + b + b^2) i

and more generally

.. math:: y_t = b y_{t-1} + i = (1+ b + b^2 + \cdots + b^t) i

or

.. math:: y_t = \frac{1-b^{t+1}}{1 -b } i 

Evidently, as :math:`t \rightarrow + \infty`,

.. math:: y_t \rightarrow \frac{1}{1-b} i 

**Remark 1:** The above formula is often applied to assert that an
exogenous increase in investment of :math:`\Delta i` at time :math:`0`
ignites a dynamic process of increases in national income by amounts

.. math:: \Delta i, (1 + b )\Delta i, (1+b + b^2) \Delta i , \cdots

 at times :math:`0, 1, 2, \ldots`

**Remark 2** Let :math:`g_t` be an exogenous sequence of government
expenditures

If we generalize the model so that the national income identity
becomes

.. math:: c_t + i_t + g_t  = y_t

then a version of the preceding argument shows that the **government
expenditures multiplier** is also :math:`\frac{1}{1-b}`, so that a
permanent increase in government expenditures ultimately leads to an
increase in national income equal to the multiplier times the increase
in government expenditures

.. Dongchen: can you think of some simple Python things to add to 
.. illustrate basic concepts, maybe the idea of a "difference equation" and how we solve it?


Interest rates and present values
---------------------------------

We can apply our formula for geometric series to study how interest
rates affect values of streams of dollar payments that extend over time

We work in discrete time and assume that :math:`t = 0, 1, 2, \ldots`
indexes time

We let :math:`r \in (0,1)` be a one-period **net nominal interest rate**

-  if the nominal interest rate is :math:`5` percent,
   then :math:`r= .05`

A one-period **gross nominal interest rate** :math:`R` is defined as

.. math:: R = 1 + r \in (1, 2) 

-  if :math:`r=.05`, then :math:`R = 1.05`

**Remark:** The gross nominal interest rate :math:`R` is an **exchange
rate** or **relative price** of dollars at between times :math:`t` and
:math:`t+1`. The units of :math:`R` are dollars at time :math:`t+1` per
dollar at time :math:`t`

When people borrow and lend, they trade dollars now for dollars later or
dollars later for dollars now

The price at which these exchanges occur is the gross nominal interest
rate

-  If I sell :math:`x` dollars to you today, you pay me :math:`R x`
   dollars tomorrow

-  This means that you borrowed :math:`x` dollars for me at a gross
   interest rate :math:`R` and a net interest rate :math:`r`

We assume that the net nominal interest rate :math:`r` is fixed over
time, so that :math:`R` is the gross nominal interest rate at times
:math:`t=0, 1, 2, \ldots`

Two important geometric sequences are

.. math:: 
  :label: geom1
  
  1, R, R^2, \cdots

and

.. math:: 
  :label: geom2
  
  1, R^{-1}, R^{-2}, \cdots

Sequence :eq:`geom1` tells us how dollar values of an investment **accumulate**
through time

Sequence :eq:`geom2` tells us how to **discount** future dollars to get their
values in terms of today's dollars

Accumulation
-------------

Geometric sequence :eq:`geom1` tells us how one dollar invested and re-invested
in a project with gross one period nominal rate of return accumulates

-  here we assume that net interest payments are reinvested in the
   project

-  thus, :math:`1` dollar invested at time :math:`0` pays interest
   :math:`r` dollars after one period, so we have :math:`r+1 = R`
   dollars at time\ :math:`1`

-  at time :math:`1` we reinvest :math:`1+r =R` dollars and receive interest
   of :math:`r R` dollars at time :math:`2` plus the *principal*
   :math:`R` dollars, so we receive :math:`r R + R = (1+r)R = R^2`
   dollars at the end of period :math:`2`

-  and so on

Evidently, if we invest :math:`x` dollars at time :math:`0` and
reinvest the proceeds, then the sequence

.. math:: x , xR , x R^2, \cdots

tells how our account accumulates at dates :math:`t=0, 1, 2, \ldots`

Discounting
------------

Geometric sequence :eq:`geom2` tells us how much future dollars are worth in
terms of today's dollars

Remember that the units of :math:`R` are dollars at :math:`t+1` per
dollar at :math:`t`

It follows that

-  the units of :math:`R^{-1}` are dollars at :math:`t` per dollar
   at\ :math:`t+1`

-  the units of :math:`R^{-2}` are dollars at :math:`t` per dollar
   at\ :math:`t+2`

-  and so on; the units of :math:`R^{-j}` are dollars at :math:`t` per
   dollar at :math:`t+j`

So if someone has a claim on :math:`x` dollars at time :math:`t+j`, it
is worth :math:`x R^{-j}` dollars at time :math:`t` (e.g., today)

Application to asset pricing
-----------------------------

A **lease** requires a payments stream of :math:`x_t` dollars at
times :math:`t = 0, 1, 2, \ldots` where

.. math::  x_t = G^t x_0 

where :math:`G = (1+g)` and :math:`g \in (0,1)`

Thus, lease payments increase at :math:`g` percent per period

For a reason soon to be revealed, we assume that :math:`G < R`

The **present value** of the lease is

.. math::

    \eqalign{ p_0  & = x_0 + x_1/R + x_2/(R^2) + \ddots \\
                     & = x_0 (1 + G R^{-1} + G^2 R^{-2} + \cdots ) \\
                     & = x_0 \frac{1}{1 - G R^{-1}} }

where the last line uses the formula for an infinite geometric series

Recall that :math:`R = 1+r` and :math:`G = 1+g` and that :math:`R > G`
and :math:`r > g` and that :math:`r` and\ :math:`g` are typically small
numbers, e.g., .05 or .03

Use the Taylor series of :math:`\frac{1}{1+r}` about :math:`r=0`,
namely,

.. math:: \frac{1}{1+r} = 1 - r + r^2 - r^3 + \cdots

and the fact that :math:`r` is small to aproximate
:math:`\frac{1}{1+r} \approx 1 - r`

Use this approximation to write :math:`p_0` as

.. math::

    \begin{align} 
    p_0 &= x_0 \frac{1}{1 - G R^{-1}} \\
    &= x_0 \frac{1}{1 - (1+g) (1-r) } \\
    &= x_0 \frac{1}{1 - (1+g - r - rg)} \\
    & \approx x_0 \frac{1}{r -g }
   \end{align}

where the last step uses the approximation :math:`r g \approx 0`

The approximation

.. math:: p_0 = \frac{x_0 }{r -g }

is known as the **Gordon formula** for the present value or current
price of an infinite payment stream :math:`x_0 G^t` when the nominal
one-period interest rate is :math:`r` and when :math:`r > g`



Notes to Dongchen
===================

Hi. We can do various things with the above formulas --

-  we can generalize them to apply to finite payment streams -- we'd
   just apply the formula for finite geometric series

-  we can do various **experiments** using both Python and calculus
   (maybe using Python to do the calculus)

   -  E.g., we can show what happens to values of assets when
      :math:`r`\ increases and when\ :math:`g` increases

   -  we can show what happens to the value of assets when
      :math:`g`\ approaches\ :math:`r` from below

   -  we can show how the formulas for finite streams still apply when
      :math:`g >r` but how they break down when :math:`r < g`

what do you think?

To do ideas -- from John Stachurski and me
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It will look nice if your students could add in figures showing, for
example, the time path of y\_t in the Keynesian multiplier. I guess you
already have this in mind.

The accumulation example could be extended, possibly in a separate
lecture, by showing the time path for household wealth when

w\_{t+1} = (1 + r) (w\_t + y\_t - c\_t)

Here c\_t could be a fixed fraction of current wealth. Or perhaps c\_t =
w\_t + y\_t at low levels of wealth but savings is positive above some
threshold.

Then one can investigate the time path of wealth under different
assumptions for {y\_t}, which is taken to be deterministic but
endogenous.

For example, how is w\_T at some large T affected by a change in the
growth rate of y\_t? How about a change in r? Which is more important?

The students will see the power of compounding in a setting they can
relate to.

They could also investigate the impact of different kinds of savings
behavior.

After that, taxes could be introduced.

As separate lectures, we could also teach linear systems when discussing
supply and demand, and then switch to nonlinear systems. Ordinarily such
topics would be though of as too hard for students at this level, but we
can use it as an opportunity to teach the bisection algorithm for
finding the zero of a function.

Even my kids are hearing the word "algorithm" at school and starting to
learn about them.

Regards, John.

On Wed, Aug 1, 2018 at 6:30 AM Thomas J Sargent thomas.sargent@nyu.edu
wrote: Dear John, As an experiment I am working with two high school
kids to try to write a low level notebook/lecture. Geometric series is
the topic. So far, all of the input has been from me on this one. I
attach the first draft of the notebook. I'll probably have to complete
this myself, maybe with some help from Natasha.

I'll try to think of a couple of other applications beyond the three I
have started with.

Tom

--

Visit me at http://johnstachurski.netm John and me

Dear John, Thanks for these great suggestions. I am going to add them as
a "to do" cell at the end of the lecture.

I agree that adding the types of graphs that you suggest will make the
lecture much better. This is the sort of thing I hope to do.

Tom

On Tue, Jul 31, 2018 at 8:11 PM John Stachurski
john.stachurski@gmail.com wrote: Hi Tom,

This is great!

It will look nice if your students could add in figures showing, for
example, the time path of y\_t in the Keynesian multiplier. I guess you
already have this in mind.

The accumulation example could be extended, possibly in a separate
lecture, by showing the time path for household wealth when

w\_{t+1} = (1 + r) (w\_t + y\_t - c\_t)

Here c\_t could be a fixed fraction of current wealth. Or perhaps c\_t =
w\_t + y\_t at low levels of wealth but savings is positive above some
threshold.

Then one can investigate the time path of wealth under different
assumptions for {y\_t}, which is taken to be deterministic but
endogenous.

For example, how is w\_T at some large T affected by a change in the
growth rate of y\_t? How about a change in r? Which is more important?

The students will see the power of compounding in a setting they can
relate to.

They could also investigate the impact of different kinds of savings
behavior.

After that, taxes could be introduced.

As separate lectures, we could also teach linear systems when discussing
supply and demand, and then switch to nonlinear systems. Ordinarily such
topics would be though of as too hard for students at this level, but we
can use it as an opportunity to teach the bisection algorithm for
finding the zero of a function.

Even my kids are hearing the word "algorithm" at school and starting to
learn about them.

Regards, John.

