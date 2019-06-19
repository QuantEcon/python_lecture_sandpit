.. _coase:

.. include:: /_static/includes/lecture_howto_py.raw

.. highlight:: python3

**************************************************************
:index:`Coase's Theory of the Firm`
**************************************************************

.. contents:: :depth: 2

Overview
============

In 1937, future Nobel Laureate Ronald Coase 
wrote a brilliant essay on the nature of the firm :cite:`coase1937nature`.

Coase was writing at a time when the Soviet Union was beginning its rise as a significant industrial power.

At the same time, many free market economies were afflicted by a severe and painful depression.

These contrasting outcomes led economists to intensively debate the relative
merits of decentralized, price based allocation versus top down planning.

In the midst of this debate, Coase made an important observation:
even in free market economies, a great deal of top-down planning does in fact take place.

This is because firms exist and, within firms, allocation is by planning.

In other words, free market economies blend both planning (within firms) and decentralized production coordinated by prices.  

The question Coase asked is this: if prices and free markets are so efficient, then why do firms even exist?

Couldn't the associated planning be done more efficiently by the market?


Why Firms Exist
----------------

On top of asking a deep and fascinating question, Coase also supplied an illuminating answer: firms exist because of transaction costs.

For example, suppose agent A runs a small business and has ongoing need for a web developer.

She can use the labor of agent B, a freelance web developer, by writing up a contract and agreeing on a suitable price.

But contracts like this can be time consuming and difficult to verify.

* How will agent A be able to specify exactly what she wants, to the finest detail, when she herself isn't sure how the business will evolve over the
coming year?

* And what if she isn't familiar with web technology?  How can she specify the details?

* And if things go badly, will failure to comply with the contract, at least in her view, be verifiable in court?


In this situation, perhaps it will be easier to *employ* agent B under a simple labor contract.

The cost of this contract is far smaller because such contracts are standard.

The basic agreement is that agent B will do what A asks him to do for the term of the contract.

This is much easier than trying to map every task out in advance in a contract that will hold up in a court of law.


A Trade-Off
--------------

Actually, we haven't yet come to the heart of Coase's investigation.

The issue of why firms exist is a binary question: should firms have positive size or zero size?

A better question is: what determines the **size of firms**, and hence the amount of conscious planning that takes place?

This is the question that Coase sought to answer.



A Quantitative Interpretation
--------------------------------





The Model
==========================

This is another section.  Here's some maths:

.. math::

    y_{t+1} := f(k_{t+1}) \xi_{t+1}

.. code-block:: python3

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline



Assumptions and Comments
---------------------------

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


