
.. _test_lecture:

.. include:: /_static/includes/lecture_howto_py.raw

.. highlight:: python3

**************************************************************
:index:`Test Lecture Title`
**************************************************************

.. contents:: :depth: 2

Overview
============

This is a test lecture

Here's some test citations: 

* :cite:`StokeyLucas1989`

* :cite:`Ljungqvist2012`

As usual, links are done like this:

* `interpolation.py package <https://github.com/EconForge/interpolation.py>`_

Here's a code block:

.. code-block:: python3

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline


The Model
==========================

This is another section.  Here's some maths:

.. math::

    y_{t+1} := f(k_{t+1}) \xi_{t+1}




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

Here is how we include code

.. literalinclude:: /_static/lecture_specific/test_lecture/test_python_file.py


New Section
============

Here's a new section.
