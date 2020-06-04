.. _multi_hyper_v7:

.. include:: /_static/includes/header.raw

.. highlight:: python3

*****************************************
Multivariate hypergeometric distribution
*****************************************

.. contents:: :depth: 2


In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon


Overview
=========

Let's Start with some imports-

.. code-block:: ipython

    import numpy as np
    from scipy.special import comb
    from scipy.stats import normaltest
    from numba import njit, prange
    import matplotlib.pyplot as plt
    %matplotlib inline
    import matplotlib.cm as cm

Assume there are in total :math:`c` types of objects in an urn. If there
are :math:`K_{i}` type :math:`i` object in the urn and you take
:math:`n` draws at random without replacement, then the number of object
of each type :math:`i` in the sample
:math:`\left(k_{1},k_{2},\dots,k_{c}\right)` has the multivariate
hypergeometric distribution. Note that :math:`N=\sum_{i=1}^{c} K_{i}` is
the total number of objects in the urn and :math:`n=\sum_{i=1}^{c}k_{i}`
must hold.

We know the following properties of multivariate hypergeometric
distribution:

Probability mass function:

.. math::


   \Pr \left(\{K_{i}=k_{i} \  \forall i\}\right) = {\displaystyle {\frac {\prod _{i=1}^{c}{\binom {K_{i}}{k_{i}}}}{\binom {N}{n}}}}

Moments:

1. mean

.. math::


   {\displaystyle \operatorname {E} (X_{i})=n{\frac {K_{i}}{N}}}

2. variance and covariance

.. math::


   {\displaystyle \operatorname {Var} (X_{i})=n{\frac {N-n}{N-1}}\;{\frac {K_{i}}{N}}\left(1-{\frac {K_{i}}{N}}\right)}

.. math::


   {\displaystyle \operatorname {Cov} (X_{i},X_{j})=-n{\frac {N-n}{N-1}}\;{\frac {K_{i}}{N}}{\frac {K_{j}}{N}}}

.. code-block:: python3

    class Urn:
    
        def __init__(self, K_arr):
            """
            Initialization given the number of each type i object in the urn.
    
            Parameters
            ----------
            K_arr: ndarray(int)
                number of each type i object.
            """
    
            self.K_arr = np.array(K_arr)
            self.N = np.sum(K_arr)
            self.c = len(K_arr)
    
        def pmf(self, k_arr):
            """
            Probability mass function.
    
            Parameters
            ----------
            k_arr: ndarray(int)
                number of observed successes of each object.
            """
    
            K_arr, N = self.K_arr, self.N
    
            k_arr = np.atleast_2d(k_arr)
            n = np.sum(k_arr, 1)
    
            num = np.prod(comb(K_arr, k_arr), 1)
            denom = comb(N, n)
    
            pr = num / denom
    
            return pr
    
        def moments(self, n):
            """
            Compute the mean and variance-covariance matrix for
            multivariate hypergeometric distribution.
    
            Parameters
            ----------
            n: int
                number of draws.
            """
    
            K_arr, N, c = self.K_arr, self.N, self.c
    
            # mean
            μ = n * K_arr / N
    
            # variance-covariance matrix
            Σ = np.ones((c, c)) * n * (N - n) / (N - 1) / N ** 2
            for i in range(c-1):
                Σ[i, i] *= K_arr[i] * (N - K_arr[i])
                for j in range(i+1, c):
                    Σ[i, j] *= - K_arr[i] * K_arr[j]
                    Σ[j, i] = Σ[i, j]
    
            Σ[-1, -1] *= K_arr[-1] * (N - K_arr[-1])
    
            return μ, Σ
        
        def simulate(self, n, size=1, seed=None):
            """
            Simulate a sample from multivariate hypergeometric
            distribution where at each draw we take n objects
            from the urn without replacement.
    
            Parameters
            ----------
            n: int
                number of objects for each draw.
            size: int(optional)
                sample size.
            seed: int(optional)
                random seed.
            """
    
            K_arr = self.K_arr
    
            gen = np.random.Generator(np.random.PCG64(seed))
            sample = gen.multivariate_hypergeometric(K_arr, n, size=size)
    
            return sample

Usage
=====

First example
-------------

Apply this to an example from
`wiki <https://en.wikipedia.org/wiki/Hypergeometric_distribution#Multivariate_hypergeometric_distribution>`__:

Suppose there are 5 black, 10 white, and 15 red marbles in an urn. If
six marbles are chosen without replacement, the probability that exactly
two of each color are chosen is

.. math::


   P(2{\text{ black}},2{\text{ white}},2{\text{ red}})={{{5 \choose 2}{10 \choose 2}{15 \choose 2}} \over {30 \choose 6}}=0.079575596816976

.. code-block:: python3

    # construct the urn
    K_arr = [5, 10, 15]
    urn = Urn(K_arr)

.. code-block:: python3

    k_arr = [2, 2, 2] # array of number of observed successes
    urn.pmf(k_arr)

If we observe for more than one time, we can construct a 2-dimensional
array ``k_arr`` and ``pmf`` will return an array of probabilities for
observing each case.

.. code-block:: python3

    k_arr = [[2, 2, 2], [1, 3, 2]]
    urn.pmf(k_arr)

Now let’s compute the mean and variance-covariance matrix.

.. code-block:: python3

    n = 6
    μ, Σ = urn.moments(n)

.. code-block:: python3

    μ

.. code-block:: python3

    Σ

Second example
--------------

Here we consider another example that Tom suggested, where the array of
number of each type :math:`i` object in the urn is
:math:`\left(157, 11, 46, 24\right)`.

.. code-block:: python3

    K_arr = [157, 11, 46, 24]
    urn = Urn(K_arr)

.. code-block:: python3

    k_arr = [10, 1, 4, 0]
    urn.pmf(k_arr)

If we observe for more than one time, we can construct a 2-dimensional
array ``k_arr`` and ``pmf`` will return an array of probabilities for
observing each case.

.. code-block:: python3

    k_arr = [[5, 5, 4 ,1], [10, 1, 2, 2], [13, 0, 2, 0]]
    urn.pmf(k_arr)

Now let’s compute the mean and variance-covariance matrix.

.. code-block:: python3

    n = 6 # number of draws
    μ, Σ = urn.moments(n)

.. code-block:: python3

    # mean
    μ

.. code-block:: python3

    # variance-covariance matrix
    Σ

We can simulate a large sample and verify the population mean and
covariance matrix we have computed above using data.

.. code-block:: python3

    size = 10_000_000
    sample = urn.simulate(n, size=size)

.. code-block:: python3

    # mean
    np.mean(sample, 0)

.. code-block:: python3

    # variance covariance matrix
    np.cov(sample.T)

In the following, we will simulate a large sample from normal
distribution using the same mean and covariance matrix and compare with
the empirical multivariate hypergeometric distribution we obtained
above.

.. code-block:: python3

    sample_normal = np.random.multivariate_normal(μ, Σ, size=size)

.. code-block:: python3

    def bivariate_normal(x, y, μ, Σ, i, j):
    
        μ_x, μ_y = μ[i], μ[j]
        σ_x, σ_y = np.sqrt(Σ[i, i]), np.sqrt(Σ[j, j])
        σ_xy = Σ[i, j]
    
        x_μ = x - μ_x
        y_μ = y - μ_y
    
        ρ = σ_xy / (σ_x * σ_y)
        z = x_μ**2 / σ_x**2 + y_μ**2 / σ_y**2 - 2 * ρ * x_μ * y_μ / (σ_x * σ_y)
        denom = 2 * np.pi * σ_x * σ_y * np.sqrt(1 - ρ**2)
    
        return np.exp(-z / (2 * (1 - ρ**2))) / denom

.. code-block:: python3

    @njit
    def count(vec1, vec2, n):
        size = sample.shape[0]
    
        count_mat = np.zeros((n+1, n+1))
        for i in prange(size):
            count_mat[vec1[i], vec2[i]] += 1
        
        return count_mat

.. code-block:: python3

    c = urn.c
    fig, axs = plt.subplots(c, c, figsize=(14, 14))
    
    # grids for ploting the bivariate Gaussian
    x_grid = np.linspace(-2, n+1, 100)
    y_grid = np.linspace(-2, n+1, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    for i in range(c):
        axs[i, i].hist(sample[:, i], bins=np.arange(0, n, 1), alpha=0.5, density=True, label='hypergeom')
        axs[i, i].hist(sample_normal[:, i], bins=np.arange(0, n, 1), alpha=0.5, density=True, label='normal')
        axs[i, i].legend()
        axs[i, i].set_title('$k_{' +str(i+1) +'}$')
        for j in range(c):
            if i == j:
                continue
    
            # bivariate Gaussian density function
            Z = bivariate_normal(X, Y, μ, Σ, i, j)
            cs = axs[i, j].contour(X, Y, Z, 4, colors="black", alpha=0.6)
            axs[i, j].clabel(cs, inline=1, fontsize=10)
    
            # empirical multivariate hypergeometric distrbution
            count_mat = count(sample[:, i], sample[:, j], n)
            axs[i, j].pcolor(count_mat.T/size, cmap='Blues')
            axs[i, j].set_title('$(k_{' +str(i+1) +'}, k_{' + str(j+1) + '})$')
    
    plt.show()

The diagonal graphs plot the marginal distributions of :math:`k_i` for
each :math:`i` using histograms, in which we observe significant
differences between hypergeometric distribution and normal distribution.

The off-diagonal graphs plot the empirical joint distribution of
:math:`k_i` and :math:`k_j` for each pair :math:`(i, j)`. The darker the
blue, the more data points are contained in the corresponding cell.
(Note that :math:`k_i` is on the x-axis and :math:`k_j` is on the
y-axis). The contour maps plot the bivariate Gaussian density function
of :math:`\left(k_i, k_j\right)` with the population mean and covariance
given by slices of :math:`\mu` and :math:`\Sigma` that we computed
above.

Let’s also test the normality for each :math:`k_i` using
``scipy.stats.normaltest`` which is based on D’Agostino and Pearson’s
test that combines skew and kurtosis to produce an omnibus test of
normality. The null hypothesis is that the sample follows normal
distribution. ``normaltest`` returns an array of p-values associated
with tests for each :math:`k_i` sample.

.. code-block:: python3

    test_multihyper = normaltest(sample)
    test_multihyper.pvalue

As we can see, all the p-values are almost :math:`0` and the null
hypotheses can be rejected.

By contrast, the sample from normal distribution does not reject the
null hypotheses.

.. code-block:: python3

    test_normal = normaltest(sample_normal)
    test_normal.pvalue

