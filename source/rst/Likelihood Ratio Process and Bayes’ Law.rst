Likelihood Ratio Process and Bayes’ Law
==========================================


Let :math:`\pi_t` be a Bayesian posterior defined as

.. math::  \pi_t = {\rm Prob}(q=f|w^t)

The likelihood ratio process is a principal actor in the formula that governs the evolution
of the posterior probability :math:`\pi_t`, an instance of **Bayes' Law**.

Bayes’ law implies that :math:`\{\pi_t\}` obeys the recursion

.. math::
   :label: eq_recur1

   \pi_t=\frac{\pi_{t-1} l_t(w_t)}{\pi_{t-1} l_t(w_t)+1-\pi_{t-1}}

with :math:`\pi_{0}` be a Bayesian prior probability that :math:`q = f`,
i.e., a belief about :math:`q` based on having seen no data.

Below we define a Python function that updates belief :math:`\pi` using
likelihood ratio :math:`\ell` according to  recursion :eq:`eq_recur1`

.. code-block:: python3

    @njit
    def update(π, l):
        "Update π using likelihood l"

        # Update belief
        π = π * l / (π * l + 1 - π)

        return π

Formula :eq:`eq_recur1` can be generalized in a useful way.

We do this by iterating on recursion :eq:`eq_recur1` in order to derive an
expression for  the time :math:`t` posterior :math:`\pi_{t+1}` as a function
of the time :math:`0` prior :math:`\pi_0` and the likelihood ratio process
:math:`L(w^{t+1})` at time :math:`t`.

To begin, notice that the updating rule

.. math::

   \pi_{t+1}
   =\frac{\pi_{t}\ell \left(w_{t+1}\right)}
   {\pi_{t}\ell \left(w_{t+1}\right)+\left(1-\pi_{t}\right)}

implies

.. math::


   \begin{aligned}
   \frac{1}{\pi_{t+1}}
       &=\frac{\pi_{t}\ell \left(w_{t+1}\right)
           +\left(1-\pi_{t}\right)}{\pi_{t}\ell \left(w_{t+1}\right)} \\
       &=1-\frac{1}{\ell \left(w_{t+1}\right)}
           +\frac{1}{\ell \left(w_{t+1}\right)}\frac{1}{\pi_{t}}.
   \end{aligned}

.. math::

   \Rightarrow
   \frac{1}{\pi_{t+1}}-1
   =\frac{1}{\ell \left(w_{t+1}\right)}\left(\frac{1}{\pi_{t}}-1\right).

Therefore

.. math::


   \begin{aligned}
       \frac{1}{\pi_{t+1}}-1
       =\frac{1}{\prod_{i=1}^{t+1}\ell \left(w_{i}\right)}
           \left(\frac{1}{\pi_{0}}-1\right)
       =\frac{1}{L\left(w^{t+1}\right)}\left(\frac{1}{\pi_{0}}-1\right).
   \end{aligned}

Since :math:`\pi_{0}\in\left(0,1\right)` and
:math:`L\left(w^{t+1}\right)>0`, we can verify that
:math:`\pi_{t+1}\in\left(0,1\right)`.

After rearranging the preceding equation, we can express :math:`\pi_{t+1}` as a
function of  :math:`L\left(w^{t+1}\right)`, the  likelihood ratio process at :math:`t+1`,
and the initial prior :math:`\pi_{0}`

.. math::
   :label: eq_Bayeslaw103

   \pi_{t+1}=\frac{\pi_{0}L\left(w^{t+1}\right)}{\pi_{0}L\left(w^{t+1}\right)+1-\pi_{0}} .

Formula :eq:`eq_Bayeslaw103` generalizes generalizes formula :eq:`eq_recur1`.

Formula :eq:`eq_Bayeslaw103`  can be regarded as a one step  revision of prior probability :math:`\pi_0` after seeing
the batch of data :math:`\left\{ w_{i}\right\} _{i=1}^{t+1}`.

Formula :eq:`eq_Bayeslaw103` shows the key role that the likelihood ratio process  :math:`L\left(w^{t+1}\right)` plays in determining
the posterior probability :math:`\pi_{t+1}`.

Formula :eq:`eq_Bayeslaw103` is the foundation for the insight that, because of the way the likelihood ratio process behaves
as :math:`t \rightarrow + \infty`, the likelihood ratio process dominates the initial prior :math:`\pi_0` in determining the
limiting behavior of :math:`\pi_t`.

To illustrate this insight, below we will plot  graphs showing **one** simulated
path of the  likelihood ratio process :math:`L_t` along with two paths of
:math:`\pi_t` that are associated with the *same* realization of the likelihood ratio process but *different* initial prior probabilities
probabilities :math:`\pi_{0}`.

First, we specify the two values of :math:`\pi_0`.

.. code-block:: python3

    π1, π2 = 0.2, 0.8

Next we generate paths of the likelihood ratio process :math:`L_t` and the posteior :math:`\pi_t` for a history drawn as IID
draws from density :math:`f`.

.. code-block:: python3

    T = l_arr_f.shape[1]
    π_seq_f = np.empty((2, T+1))
    π_seq_f[:, 0] = π1, π2

    for t in range(T):
        for i in range(2):
            π_seq_f[i, t+1] = update(π_seq_f[i, t], l_arr_f[0, t])

.. code-block:: python3

    fig, ax1 = plt.subplots()

    for i in range(2):
        ax1.plot(range(T+1), π_seq_f[i, :], label=f"$\pi_0$={π_seq_f[i, 0]}")

    ax1.set_ylabel("$\pi_t$")
    ax1.set_xlabel("t")
    ax1.legend()
    ax1.set_title("when f governs data")

    ax2 = ax1.twinx()
    ax2.plot(range(1, T+1), np.log(l_seq_f[0, :]), '--', color='b')
    ax2.set_ylabel("$log(L(w^{t}))$")

    plt.show()


The dotted line in the graph above records the logarithm of the  likelihood ratio process :math:`\log L(w^t)`.


Please note that there are two different scales on the :math:`y` axis.

Now let's study what happens when the history consists of IID draws from density :math:`g`


.. code-block:: python3

    T = l_arr_g.shape[1]
    π_seq_g = np.empty((2, T+1))
    π_seq_g[:, 0] = π1, π2

    for t in range(T):
        for i in range(2):
            π_seq_g[i, t+1] = update(π_seq_g[i, t], l_arr_g[0, t])

.. code-block:: python3

    fig, ax1 = plt.subplots()

    for i in range(2):
        ax1.plot(range(T+1), π_seq_g[i, :], label=f"$\pi_0$={π_seq_g[i, 0]}")

    ax1.set_ylabel("$\pi_t$")
    ax1.set_xlabel("t")
    ax1.legend()
    ax1.set_title("when g governs data")

    ax2 = ax1.twinx()
    ax2.plot(range(1, T+1), np.log(l_seq_g[0, :]), '--', color='b')
    ax2.set_ylabel("$log(L(w^{t}))$")

    plt.show()


Below we offer Python code that verifies this in a setting in which
nature chose permanently to draw from density :math:`f`.

.. code-block:: python3

    π_seq = np.empty((2, T+1))
    π_seq[:, 0] = π1, π2

    for i in range(2):
        πL = π_seq[i, 0] * l_seq_f[0, :]
        π_seq[i, 1:] = πL / (πL + 1 - π_seq[i, 0])

.. code-block:: python3

    np.abs(π_seq - π_seq_f).max() < 1e-10

    Having seen how the likelihood ratio process is a key ingredient of the formula :eq:`eq_Bayeslaw103` for
a Bayesian's posteior probabilty that nature has drawn history :math:`w^t` as repeated draws from density
:math:`g`, 