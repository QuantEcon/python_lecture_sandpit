.. _coase:

.. include:: /_static/includes/lecture_howto_py.raw

.. highlight:: python3

**************************************************************
:index:`Coase's Theory of the Firm`
**************************************************************

.. contents:: :depth: 2

Overview
============

In 1937, Ronald Coase wrote a brilliant essay on the nature of the firm (see :cite:`coase1937nature`).

Coase was writing at a time when the Soviet Union was beginning its rise as a significant industrial power.

At the same time, many free market economies were afflicted by a severe and painful depression.

These contrasting outcomes led economists into an intensive debate on the relative
merits of decentralized, price based allocation versus top down planning.

In the midst of this debate, Coase made an important observation:
even in free market economies, a great deal of top-down planning does in fact take place.

This is because firms form one part of free market economies and, within firms, allocation is by planning.

In other words, free market economies blend both planning (within firms) and decentralized production coordinated by prices.  

The question Coase asked is this: if prices and free markets are so efficient, then why do firms even exist?

Couldn't the associated within-firm planning be done more efficiently by the market?


Why Firms Exist
----------------

On top of asking a deep and fascinating question, Coase also supplied an illuminating answer: firms exist because of transaction costs.

There are many examples of such costs but here is one that seems pervasive.

Suppose agent A is considering setting up a small business and needs a web developer to construct and help run an online store.

She can use the labor of agent B, a web developer, by writing up a freelance contract for these tasks and agreeing on a suitable price.

But contracts like this can be time consuming and difficult to verify.

* How will agent A be able to specify exactly what she wants, to the finest detail, when she herself isn't sure how the business will evolve over the
coming years?

* And what if she isn't familiar with web technology?  How can she specify all the relevant details?

* And, if things go badly, will failure to comply with the contract be verifiable in court?

In this situation, perhaps it will be easier to *employ* agent B under a simple labor contract.

The cost of this contract is far smaller because such contracts are simpler and more standard.

The basic agreement in a labor contract is: agent B will do what A asks him to do for the term of the contract, in return for a given salary.

This is much easier than trying to map every task out in advance in a contract that will hold up in a court of law.

So agent A decides to hire agent B and a firm of nontrivial size appears, due to transaction costs.


A Trade-Off
--------------

Actually, we haven't yet come to the heart of Coase's investigation.

The issue of why firms exist is a binary question: should firms have positive size or zero size?

A better and more general question is: **what determines the size of firms**?

The answer Coase came up with was that "a firm will tend to expand until the costs of organizing an extra
transaction within the firm become equal to the costs of carrying out the same
transaction by means of an exchange on the open market..." (:cite:`coase1937nature`, p. 395).

But what are these internal and external costs?

In short, Coase envisaged a trade-off between

* transaction costs, which add to the expense of operating *between* firms, and

* diminishing returns to management, which adds to the expense of operating *within* firms.


We discussed an example of transaction costs above (contracts).

The other cost, diminishing returns to management, is a catch-all for the idea
that big operations are increasingly costly to manage.

For example, you could think of management as a pyramid, so hiring more workers to implement more tasks
requires expansion of the pyramid, and hence labor costs grow at a rate more than
proportional to the range of tasks.

Diminishing returns to management makes in-house production expensive, favoring small firms.  


Summary
----------

Thus, firms grow because transaction costs encourage them to take some operations in house.

But as they get large, in-house operations become costly too.

The size of firms is determined by balancing these effects, thereby equalizing the marginal costs of each form of operation.



A Quantitative Interpretation
--------------------------------

Coases ideas were expressed purely verbally, without any mathematics.

In fact his essay is a wonderful example of how far you can get with clear thinking and plain English.

However, plain English is not good for quantitative analysis, or for plugging into a computer.

So let's bring some mathematical and computation tools to bear.

In doing so we'll add a bit more struture than Coase did, but this price will be worth paying.


Our exposition is based on :cite:`kikuchi2018span`.


The Model
==========================


We study production of a single unit of a final good. 

There is a linearly ordered production chain, where the good is produced through
the sequential completion of a large number of processing stages

The stages are indexed by :math:`t \in [0,1]`, with :math:`t=0` indicating that no tasks
have been undertaken and :math:`t=1` indicating that the good is complete.  

Subcontracting
------------------

The subcontracting scheme by which tasks are allocated across firms is illustrated in the figure below.

[add fig here]

In this example,

* Firm 1 receives a contract to sell one unit of the completed good to a final buyer.  
  
* Firm 1 then forms a contract with firm 2 to purchase the partially completed good at stage :math:`t_1`, with the intention of implementing the
remaining $1 - t_1$ tasks in-house (i.e., processing from stage $t_1$ to stage
$1$).  

* Firm~2 repeats this procedure, forming a contract with firm~3 to purchase the good at stage $t_2$.  

* firm~3 decides to complete the chain, selecting $t_3 = 0$.  

At this point, production unfolds in the opposite direction (i.e., from upstream to downstream).  

* Firm 3 completes processing stages from $t_3 = 0$ up to $t_2$ and transfers the good to firm 2.  
  
* Firm 2 then processes from $t_2$ up to $t_1$ and transfers the good to firm 1, 
  
* Firm 1 processes from $t_1$ to $1$ and delivers the completed good to the final buyer.

The length of the interval of stages (range of tasks) carried out by firm $i$ is denoted by $\ell_i$

[add fig 2 here]

Each firm chooses only its upstream boundary, treating its downstream boundary as given.

The benefit of this formulation is that it implies a recursive structure for the decision problem for each firm.

In choosing how many processing stages to subcontract, each successive firm faces essentially the same decision problem as the firm above it in the chain, with the only difference being that the decision space is a subinterval of the decision space for the firm above.  

We exploit this recursive structure in our study of equilibrium.



Summary of Costs
-----------------------------

Diminishing returns to management means rising costs per task when a firm expands the range of productive activities coordinated by its managers.  

We represent these ideas by taking the cost of carrying out :math:`\ell` tasks in-house to be :math:`c(\ell)`, where :math:`c` is increasing and strictly convex.   

Thus, average cost per task rises with the range of tasks performed in-house.

We also assume that :math:`c` is continuously differentiable, with :math:`c(0)=0` and :math:`c'(0) > 0`.

Transaction costs are represented as a wedge between the buyer's and seller's prices. 

It matters little for us whether the transaction cost is borne by the buyer or the seller.  

Here we assume that the cost is borne only by the buyer.  

In particular, when two firms agree to a trade at face value :math:`v`, the buyer's total outlay is :math:`\delta v`, where :math:`\delta > 1`.  

The seller receives only :math:`v`, and the difference is paid to agents outside the model.






Assumptions and Comments
---------------------------


This is another section.  Here's some maths:

.. math::

    y_{t+1} := f(k_{t+1}) \xi_{t+1}

.. code-block:: python3

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline



This is a subsection.  


Optimization
^^^^^^^^^^^^^^^

This is a sub-subsection.

Here's some maths with a label:

.. math::
    :label: texs0_og2

    \mathbb E \left[ \sum_{t = 0}^{\infty} \beta^t u(c_t) \right]


We cite it like this: Equation :eq:`texs0_og2` 

Here's a numbered list

#. point one

#. point two


