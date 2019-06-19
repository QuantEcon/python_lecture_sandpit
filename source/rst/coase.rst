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

This is the question that Coase sought to answer.

His insight


A Quantitative Interpretation
--------------------------------

Coases ideas were expressed purely verbally, without any mathematics.

In fact his essay is a great example of how far you can go with clear thinking and plain English.

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


