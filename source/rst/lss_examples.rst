
Methods of the Linear State Space Class
=======================================

QuantEcon contains a class which has various methods for operating on
Linear State Space Models (such models are explained in detail `in this
lecture <https://lectures.quantecon.org/py/linear_models.html>`__)

In this notebook, we will illustrate the methods that can be used on
instances of this class

To illustrate these methods, we will use Paul Samuleson's (1939)
multiplier-accelerator model

Samuelson's (1939) multiplier-accelerator model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The previous lecture showed (among other things) how to represent a
univariate auto-regressive processes as a linear state space model. We
will do the same for Samuelson's (1939) multiplier-accelerator model

Assume that

-  :math:`\{G_t\}` is a sequence of levels of government expenditures

-  :math:`\{C_t\}` is a sequence of levels of aggregate consumption
   expenditures, a key endogenous variable in the model

-  :math:`\{I_t\}` is a sequence of rates of investment, another key
   endogenous variable

-  :math:`\{Y_t\}` is a sequence of levels of national income, yet
   another endogenous variable

-  :math:`a` is the "marginal propensity to consume" in the Keynesian
   consumption function :math:`C_t = a Y_{t-1} + \gamma`

-  :math:`b` is the "accelerator coefficient" in the "investment
   accerator" $I\_t = b (Y\_{t-1} - Y\_{t-2}) $

-  :math:`\{\epsilon_{t}\}` is a sequence of independently and
   identically distributed :math:`N(0,1)` random variables, i.e., mean
   zero, variance one

-  :math:`\sigma` :math:`\geq` 0 is a "volatility"
   parameter.

The model combines the consumption function

.. math::  C_t = a Y_{t-1} + \gamma  \quad \quad (1) 

with the \`\`investment accelerator''

.. math::  I_t = b (Y_{t-1} - Y_{t-2}) + \sigma \epsilon_{t} \quad \quad (2) 

and the national income identity

.. math::  Y_t = C_t + I_t + G_t \quad \quad (3) 

Equations (1), (2), and (3) imply the following second-order
auto-regressive process for national income:

.. math::  Y_t = (a+b) Y_{t-1} - b Y_{t-2} + (\gamma + G_t)  + \sigma \epsilon_t 

If we assume that :math:`G_t = G \, \forall \, t`, this can be written
as a linear state space model in the following way:

.. math::

    \eqalign{ \left[\matrix{Y_{t+1} \cr
                    Y_t \cr
                     1 \cr
                     I_{t+1} \cr
                     } \right] &=
       \left[\matrix{a + b & -b & \gamma + G & 0 \cr
                      1 & 0 & 0 & 0 \cr
                      0 &0 & 1 & 0\cr
                      b & -b & 0 & 0 \cr}\right]
       \left[\matrix{Y_t \cr
                     Y_{t-1} \cr
                      1 \cr
                      I_t \cr} \right]
       + \left[\matrix{ \sigma \cr
                        0 \cr
                0 \cr
                \sigma \cr} \right] w_{t+1}  \cr
        \left[\matrix{
                    Y_t \cr
                    C_t \cr
                    I_t \cr
                     } \right]  & = \left[\matrix{1 & 0 & 0 & 0 \cr
                                                  0 & a & \gamma & 0\cr
                                                  0 & 0 & 0 & 1\cr} \right] 
       \left[\matrix{Y_t \cr
                     Y_{t-1} \cr
                      1 \cr
                      I_t \cr} \right]}  

We use this form as we are interested in the paths of :math:`C` and
:math:`I` as well as :math:`Y`

We want to ask the following questions of this model: 

\* Does :math:`Y_t` have a stationary distribution? If so, what is it? 
\* How long does it take for :math:`Y_t` to reach its stationary distribution?
\* What is the effect of an investment shock, :math:`\sigma \epsilon_t`,
on present and future values of :math:`C_t,I_t` and :math:`Y_t`?

The methods of the Linear State Space class will be able to help us out

To start, lets assume the following parameter values:

.. math::  a = 0.9,b = 0.5, \gamma = 10,\sigma = 0.5,G = 5 

.. code:: ipython3

    import numpy as np
    import matplotlib.pyplot as plt
    from quantecon import LinearStateSpace
    from scipy.stats import norm
    %matplotlib inline

.. code:: ipython3

    a, b, gamma, sigma, g = 0.95, 0.5, 10, 0.5, 5
    
    A = [[a+b, -b, gamma+g, 0],
         [1,   0,  0,       0],
         [0,   0,  1,       0],
         [b,   -b, 0,       0]]
    C = [[sigma],
         [0],
         [0],
         [sigma]]
    G = [[1, 0, 0,     0],
         [0, a, gamma, 0],
         [0, 0, 0,     1]]
    
    mu = [200, 200, 1,0]

Lets use these matrices to create an instance of the Linear State Space
class called ``Samuelson``

.. code:: ipython3

    Samuelson = LinearStateSpace(A, C, G, mu_0 = mu)

Notice, we didn't give the class a value for :math:`\Sigma_0`. In this
case, the class automatically assumes that :math:`\Sigma_0 = 0`, i.e.
the first period's values of the state vector are given by
:math:`\mu_0`

The first method we will highlight is ``simulate()``

Below we simulate :math:`C_t`, :math:`I_t` and :math:`Y_t` for 150
periods

There is evidence in the plots of the "investment accelerator".
Investment is highest in the first twenty periods, while :math:`Y_t` is
growing

.. code:: ipython3

    x, y = Samuelson.simulate(ts_length = 150)
    
    plt.figure(figsize=(12,4))
    
    plt.subplot(121)
    plt.plot(y[0:2,:].T)
    plt.xlabel('t')
    plt.legend(['$Y_t$', '$C_t$'],loc = 'lower right', fontsize=14)
    plt.title('Simulation of $C_t$ and $Y_t$', fontsize=14);
    
    plt.subplot(122)
    plt.plot(y[2,:])
    plt.xlabel('t')
    plt.title('Simulation of $I_t$', fontsize=14);

Next, we plot 200 independent simulations of :math:`\{Y_t\}` for 150
periods, each starting from :math:`Y_0 = Y_{-1} = 200`

We can see that it does appear that the model approaches a stationary
distribution, but that it takes around 50 periods to get there

.. code:: ipython3

    plt.figure(figsize=(12,4))
    for i in range(200):
        x, y = Samuelson.simulate(ts_length = 150)
        plt.plot(y[0,:])
    plt.xlabel('t')
    plt.title('200 simulations of $Y_t$ with $b = 0.5$', fontsize=14);

The next method we will use is ``stationary_distributions()``

This method starts from the initial distribution
:math:`(\mu_0,\Sigma_0)` and iterates on the following two equations
until the distribution converges (if it does so):

.. math::  \mu_{t+1} = A \mu_t

.. math::  \Sigma_{t+1} = A \Sigma_t A' + CC' 

A stationary distribution for :math:`x` is
:math:`(\mu_\infty, \Sigma_\infty)` satisfying:

.. math::  \mu_\infty = A \mu_\infty

.. math::  \Sigma_\infty = A \Sigma_\infty A' + CC' 

A stationary distribution for :math:`y` is then given by
:math:`(G \mu_\infty, G \Sigma_\infty G')`

.. code:: ipython3

    mux, muy, sigx, sigy = Samuelson.stationary_distributions()

.. code:: ipython3

    print(mux)

.. code:: ipython3

    print(sigx)

The calculation of the stationary distribution actually relies on
another method for the Linear State Space class: ``moment_sequence()``

This method is an example of a generator function

.. code:: ipython3

    gen = Samuelson.moment_sequence()
    type(gen)

We can use this generator to calculate successive values of
:math:`\mu_t, \Sigma_t`.

The first time we use the ``next()`` method on the generator, we are
given :math:`\mu_0, \Sigma_0`:

.. code:: ipython3

    mu_x0, mu_y0, sig_x0, sig_y0 = next(gen)

.. code:: ipython3

    print(mu_x0)

.. code:: ipython3

    print(sig_x0)

If we apply the ``next()`` method again, we get :math:`\mu_1, \Sigma_1`,
and so on:

.. code:: ipython3

    mu_x1, mu_y1, sig_x1, sig_y1 = next(gen)

.. code:: ipython3

    print(mu_x1)

.. code:: ipython3

    print(sig_x1)

You can read more about the benefits of generators
`here <https://lectures.quantecon.org/py/python_advanced_features.html#paf-generators>`__.

Lets return to our question about how long it takes to approach the
stationary distribution. A useful method for answering this question is
``replicate()``

This method starts from :math:`x_0 \sim N(\mu_0, \Sigma_0)` and
simulates ``num_reps`` different observations of the model for a
particular value of :math:`T` (i.e. it only returns observations of
:math:`x_T` and :math:`y_T` rather than the whole sequence)

For a large enough value of ``num_reps``, we can use this method, and
our knowledge of the stationary distribution to check how long it takes
the population moments of the model to approach the stationary
distribution

If we try T = 20, we can see that the the histogram of :math:`Y_{20}` is
not the same as the stationary distribution

.. code:: ipython3

    xT,yT = Samuelson.replicate(T=20,num_reps=10000)
    
    plt.figure(figsize=(8,4))
    plt.hist(yT[0,:], bins='auto',normed = True);
    x_axis = np.arange(mux[0] - 15, mux[0] + 15, 0.5)
    plt.plot(x_axis, norm.pdf(x_axis,mux[0][0],sigx[0][0]**0.5),label='Stationary Density')
    plt.legend(loc='best')
    plt.title('Comparing stationary density with simulations of $Y_{20}$', fontsize=14);

But it appears to be very close when :math:`T = 50`, as we might have
expected from our first simulations

.. code:: ipython3

    xT,yT = Samuelson.replicate(T=50,num_reps=10000)
    
    plt.figure(figsize=(8,4))
    plt.hist(yT[0,:], bins='auto',normed = True);
    x_axis = np.arange(mux[0] - 15, mux[0] + 15, 0.5)
    plt.plot(x_axis, norm.pdf(x_axis,mux[0][0],sigx[0][0]**0.5),label='Stationary Density')
    plt.legend(loc='best')
    plt.title('Comparing stationary density with simulations of $Y_{50}$', fontsize=14);

Now, lets consider the impact of an "investment shock" on the paths of
:math:`C_t,I_t` and :math:`Y_t`. To do this, we can use the
``impulse_response()`` method

Consider a linear state space model:

.. math::  x_{t+1} = A x_t + C w_{t+1} 

.. math::  y_t = G x_t 

By iterating on this system, we see that the impact of a vector of
shocks :math:`w_{t+1}` on :math:`x_{t+1}, x_{t+2}, x_{t+3}...` is given
by :math:`C, AC, A^2C...`

The impact on current and future values of :math:`y` is
:math:`GC, GAC, GA^2C,...`

The ``impulse_response()`` method returns these sequences of
coefficients, where :math:`j` is the number of periods that we are
interested in

.. code:: ipython3

    x_coef, y_coef = Samuelson.impulse_response(j=20)

Using these coefficients, we can plot the responses of each variable to
a one standard-deviation investment shock in our model

.. code:: ipython3

    plt.figure(figsize=(8,4))
    plt.plot(np.asarray(y_coef)[:,:,0])
    plt.xlabel('$j$',fontsize=18)
    plt.ylim([0,1])
    plt.legend(['$Y_{t+j}$', '$C_{t+j}$','$I_{t+j}$'],loc = 'upper right', fontsize=14)
    plt.title('Impulse response to positive investment shock with $b = 0.5$', fontsize=14);

Now consider what happens if we turn off the accelerator mechanism, by
setting :math:`b = 0`

Without the accelerator mechanism, the response of national income to an
investment shock is smaller, and doesn't display the "hump-shape" seen
above.

.. code:: ipython3

    b = 0
    
    A2 = [[a+b, -b, gamma+g, 0],
         [1,   0,  0,       0],
         [0,   0,  1,       0],
         [b,   -b, 0,       0]]
    
    Samuelson2 = LinearStateSpace(A2, C, G, mu_0 = mu)
    
    x_coef, y_coef = Samuelson2.impulse_response(j=20)
    
    plt.figure(figsize=(8,4))
    plt.plot(np.asarray(y_coef)[:,:,0])
    plt.xlabel('$j$',fontsize=18)
    plt.ylim([0,1])
    plt.legend(['$Y_{t+j}$', '$C_{t+j}$','$I_{t+j}$'],loc = 'upper right', fontsize=14)
    plt.title('Impulse response to positive investment shock with $b = 0$', fontsize=14);

Finally, lets consider a third parameterization, raising :math:`b` from
0.5 to 1. This means that investment now rises one-for-one with the
lagged change in national income

.. code:: ipython3

    b = 1
    
    A3 = [[a+b, -b, gamma+g, 0],
         [1,   0,  0,       0],
         [0,   0,  1,       0],
         [b,   -b, 0,       0]]
    
    Samuelson3 = LinearStateSpace(A3, C, G, mu_0 = mu)

If we try to find the stationary distribution for this new
parameterization we find that we receive an error

.. code:: ipython3

    # Samuelson3.stationary_distributions()

Simulating the model shows why; national income now displays oscillatory
behaviour

.. code:: ipython3

    x,y = Samuelson3.simulate(ts_length = 150)
    
    plt.figure(figsize=(12,4))
    for i in range(200):
        x, y = Samuelson3.simulate(ts_length = 150)
        plt.plot(y[0,:])
    plt.xlabel('t')
    plt.title('200 simulations of $Y_t$ with $b = 1$', fontsize=14);

We could have predicted this if we remembered the math of second-order
auto-regressive processes

Let :math:`z_t` follow an AR(2) process:

.. math::  z_{t+1} = \alpha + \rho_1 z_t+ \rho_2 z_{t-1} + w_{t+1} 

The following picture (borrowed from p. 189 of Macroeconomic Theory, 2nd
edition, by Thomas Sargent) shows the dynamics of :math:`z_t` that we
can expect for different values of :math:`\rho_1, \rho_2`

The red dot indicates our current set of parameters. By setting
:math:`b = 1` in the Samuelson model, we were setting
:math:`\rho_2 = -1` in the equivalent AR(2) process, and consequently
our model is on the knife-edge case between dampened and explosive
oscillations

.. code:: ipython3

    def param_plot():
    
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_aspect('equal')
    
        # Set axis
        xmin, ymin = -3, -2
        xmax, ymax = -xmin, -ymin
        plt.axis([xmin, xmax, ymin, ymax])
    
        # Set axis labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r'$\rho_2$', fontsize=16)
        ax.xaxis.set_label_position('top')
        ax.set_ylabel(r'$\rho_1$', rotation=0, fontsize=16)
        ax.yaxis.set_label_position('right')
    
        # Draw (t1, t2) points
        rho1 = np.linspace(-2, 2, 100)
        ax.plot(rho1, -abs(rho1) + 1, c='black')
        ax.plot(rho1, np.ones_like(rho1) * -1, c='black')
        ax.plot(rho1, -(rho1**2 / 4), c='black')
    
        # Turn normal axes off
        for spine in ['left', 'bottom', 'top', 'right']:
            ax.spines[spine].set_visible(False)
    
        # Add arrows to represent axes
        axes_arrows = {'arrowstyle': '<|-|>', 'lw': 1.3}
        ax.annotate('', xy=(xmin, 0), xytext=(xmax, 0), arrowprops=axes_arrows)
        ax.annotate('', xy=(0, ymin), xytext=(0, ymax), arrowprops=axes_arrows)
    
        # Annotate the plot with equations
        plot_arrowsl = {'arrowstyle': '-|>', 'connectionstyle': "arc3, rad=-0.2"}
        plot_arrowsr = {'arrowstyle': '-|>', 'connectionstyle': "arc3, rad=0.2"}
        ax.annotate(r'$\rho_1 + \rho_2 < 1$', xy=(0.5, 0.3), xytext=(0.8, 0.6),
                    arrowprops=plot_arrowsr, fontsize='12')
        ax.annotate(r'$\rho_1 + \rho_2 = 1$', xy=(0.38, 0.6), xytext=(0.6, 0.8),
                    arrowprops=plot_arrowsr, fontsize='12')
        ax.annotate(r'$\rho_2 < 1 + \rho_1$', xy=(-0.5, 0.3), xytext=(-1.3, 0.6),
                    arrowprops=plot_arrowsl, fontsize='12')
        ax.annotate(r'$\rho_2 = 1 + \rho_1$', xy=(-0.38, 0.6), xytext=(-1, 0.8),
                    arrowprops=plot_arrowsl, fontsize='12')
        ax.annotate(r'$\rho_2 = -1$', xy=(1.5, -1), xytext=(1.8, -1.3),
                    arrowprops=plot_arrowsl, fontsize='12')
        ax.annotate(r'${\rho_1}^2 + 4\rho_2 = 0$', xy=(1.15, -0.35),
                    xytext=(1.5, -0.3), arrowprops=plot_arrowsr, fontsize='12')
        ax.annotate(r'${\rho_1}^2 + 4\rho_2 < 0$', xy=(1.4, -0.7),
                    xytext=(1.8, -0.6), arrowprops=plot_arrowsr, fontsize='12')
    
        # Label categories of solutions
        ax.text(1.5, 1, 'Explosive\n growth', ha='center', fontsize=16)
        ax.text(-1.5, 1, 'Explosive\n oscillations', ha='center', fontsize=16)
        ax.text(0.05, -1.5, 'Explosive oscillations', ha='center', fontsize=16)
        ax.text(0.09, -0.5, 'Damped oscillations', ha='center', fontsize=16)
    
        # Add small marker to y-axis
        ax.axhline(y=1.005, xmin=0.495, xmax=0.505, c='black')
        ax.text(-0.12, -1.12, '-1', fontsize=10)
        ax.text(-0.12, 0.98, '1', fontsize=10)
        
        # Add point showing current parameters
        ax.scatter(a+b, -b, 80, 'red', 'o')
        
        return fig
    
    param_plot()
    plt.show()

We could also have seen this by calculating the eigenvalues of the A
matrix. The following function checks the eigenvalues of the A matrix of
a linear state space model. If all eigenvalues of A have moduli strictly
less than unity (apart from one associated with a constant in the state
vector), then the function reports that a stationary distribution
exists

The function reports that a stationary distribution exists for our first
and second sets of parameter values, but not when :math:`b` has been
reduced to :math:`-1`

.. code:: ipython3

    def A_test(A,C):
        # Find dimension of A matrix
        A = np.asarray(A)
        C = np.asarray(C)
        dim_x = A.shape[0]
        dim_w = C.shape[1]
        
        # Detect location of constant in the state vector (if it exists)
        cons_ind = []
        for j in range(dim_x):
            if np.array_equal(A[j,:] - np.eye(dim_x)[j,:],np.zeros(dim_x)) == True:
                if np.array_equal(C[j,:] - np.zeros(dim_w),np.zeros(dim_w)) == True:
                    cons_ind = j
        
        # If constant exists, create submatrix of A without constant
        if type(cons_ind) is int:
            A = np.delete(A, cons_ind, axis=0)
            A = np.delete(A, cons_ind, axis=1)
        
        # Test eigenvalues
        d,v = np.linalg.eig(A)
        if max(np.abs(d)) >= 1:
            print('Stationary distribution does not exist')
        else:
            print('Stationary distribution exists')

.. code:: ipython3

    A_test(A,C)

.. code:: ipython3

    A_test(A2,C)

.. code:: ipython3

    A_test(A3,C)

There's one final method of the Linear State Space class that we haven't
yet used.

Suppose we are interested in forecasting the following geometric sums:

.. math::  S_x = E \Big[ \sum_{j=0}^{\infty} \beta^j x_{t+j} | x_t \Big] 

.. math::  S_y = E \Big[ \sum_{j=0}^{\infty} \beta^j y_{t+j} | x_t \Big] 

In a linear state space model, these expectations are given by:

.. math::  S_x = (I - \beta A)^{-1} x_t 

.. math::  S_y = G(I - \beta A)^{-1} x_t 

We can calculate that using the ``geometric_sums()`` method.

.. code:: ipython3

    S_x1, S_y1 = Samuelson.geometric_sums(beta = 0.95, x_t = mu)
    print(S_y1)

.. code:: ipython3

    S_x2, S_y2 = Samuelson2.geometric_sums(beta = 0.95, x_t = mu)
    print(S_y2)

.. code:: ipython3

    S_x3, S_y3 = Samuelson3.geometric_sums(beta = 0.95, x_t = mu)
    print(S_y3)
