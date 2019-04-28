.. _hs_invertibility_example:

.. include:: /_static/includes/header.raw

.. highlight:: python3

****************************
Shock Non Invertibility 
****************************

.. contents:: :depth: 2


In addition to what's in Anaconda, this lecture uses the quantecon library

.. code-block:: ipython
  :class: hide-output

  !pip install quantecon

We'll make these imports

.. code-block:: ipython

    import numpy as np
    import matplotlib.pyplot as plt
    from quantecon import LQ
    from quantecon import DLE
    %matplotlib inline

This lecture can be viewed as introducing  an early contribution to what is now often called 
a **news and noise** issue

In particular, it analyzes and illustrates an **invertibility** issue that is
endemic within a class of permanent income models

Technically, the invertibility problem indicates a situation in which
histories of the shocks in an econometrician's autoregressive or Wold
moving average representation span a smaller information space than do
the shocks seen by the agent inside the econometrician's model

This situation sets the stage for an econometrician who is unaware of the
problem to  misinterpret  shocks and likely responses to them

.. code:: ipython3

    import numpy as np
    import quantecon as qe
    import matplotlib.pyplot as plt
    from quantecon import LQ
    from quantecon import DLE
    from math import sqrt
    %matplotlib inline

We consider the following modification  of Robert Hall's (1978) model :cite:`Hall1978` in which the
endowment process is the sum of two orthogonal autoregressive processes:

**Preferences**

.. math::  -\frac{1}{2}\mathbb{E}\sum_{t=0}^\infty \beta^t[(c_t - b_t)^2 + l_t^2]|J_0 

.. math:: s_t = c_t

.. math:: b_t = U_bz_t

**Technology**

.. math::  c_t + i_t = \gamma k_{t-1} + d_t

.. math::  k_t = \delta_k k_{t-1} + i_t 

.. math::  g_t = \phi_1 i_t \, , \phi_1 > 0 

.. math::  g_t \cdot g_t = l_t^2 

**Information**

.. math::

    z_{t+1} = 
   \left[ {\begin{array}{cccccc}
      1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0.9 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0
      \end{array} } 
      \right]
      z_t + 
       \left[ {\begin{array}{cc}
      0 & 0 \\ 1 & 0 \\ 0 & 4 \\ 0 & 0 \\ 0 & 0 \\ 0 & 0
      \end{array} } 
      \right]
      w_{t+1} 

.. math::

    U_b = 
      \left[ {\begin{array}{cccccc}
      30 & 0 & 0 & 0 & 0 & 0
      \end{array} } 
      \right]

.. math::

   U_d = 
      \left[ {\begin{array}{cccccc}
      5 & 1 & 1 & 0.8 & 0.6 & 0.4 \\ 0 & 0 & 0 & 0 & 0 & 0
      \end{array} } 
      \right]

The preference shock is constant at 30, while the endowment process is
the sum of a constant and two orthogonal processes.

Specifically:

.. math::  d_t = 5 + d_{1t} + d_{2t} 

.. math::  d_{1t} = 0.9d_{1t-1} + w_{1t} 

.. math::  d_{2t} = 4w_{2t} + 0.8(4w_{2t-1})+ 0.6(4w_{2t-2})+ 0.4(4w_{2t-3}) 

:math:`d_{1t}` is a first-order AR process, while :math:`d_{2t}` is a
third order pure moving average process.

.. code:: ipython3

    gamma1 = 0.05
    gamma = np.array([[gamma1],[0]])
    phic = np.array([[1],[0]])
    phig = np.array([[0],[1]])
    phi1 = 0.00001
    phii = np.array([[1],[-phi1]])
    deltak = np.array([[1]])
    thetak = np.array([[1]])
    beta = np.array([[1/1.05]])
    llambda = np.array([[0]])
    pih = np.array([[1]])
    deltah = np.array([[.9]])
    thetah = np.array([[1]])-deltah
    ud = np.array([[5,1,1,0.8,0.6,0.4],[0,0,0,0,0,0]])
    a22 = np.zeros((6,6))
    a22[[0, 1, 3, 4, 5], [0, 1, 2, 3, 4]] = np.array([1.0, 0.9, 1.0, 1.0, 1.0]) #Chase's great trick
    c2= np.zeros((6,2))
    c2[[1,2],[0,1]] = np.array([1.0,4.0])
    ub=np.array([[30,0,0,0,0,0]])
    x0 = np.array([[5],[150],[1],[0],[0],[0],[0],[0]])
    
    Info1 = (a22,c2,ub,ud)
    Tech1 = (phic,phig,phii,gamma,deltak,thetak)
    Pref1 = (beta,llambda,pih,deltah,thetah)

.. code:: ipython3

    Econ1 = DLE(Info1, Tech1, Pref1)

We define the household's net of interest deficit as :math:`c_t - d_t`.

Hall's model imposes "expected present-value budget balance" in the
sense that

.. math:: \mathbb{E}\sum_{j=0}^\infty \beta^j (c_{t+j} - d_{t+j})|J_t = \beta^{-1}k_{t-1} \, \forall t 

If we define the moving average representation of
:math:`(c_t, c_t - d_t)` in terms of the :math:`w_t`\ s to be:

.. math::

    \left[ {\begin{array}{c}
      c_t \\ c_t - d_t
      \end{array} } 
      \right]  = \left[ {\begin{array}{c}
      \sigma_1(L) \\ \sigma_2(L)
      \end{array} } 
      \right] w_t

then Hall's model imposes the restriction
:math:`\sigma_2(\beta) = [0\,\,\,0]`.

The agent inside this model sees histories of both components of the
endowment process :math:`d_{1t}` and :math:`d_{2t}`.

The econometrician has data on the history of the pair
:math:`[c_t,d_t]`, but not directly on the history of :math:`w_t`.

The econometrician obtains a Wold representation for the process
:math:`[c_t,c_t-d_t]`:

.. math::

    \left[ {\begin{array}{c}
      c_t \\ c_t - d_t
      \end{array} } 
      \right]  = \left[ {\begin{array}{c}
      \sigma_1^*(L) \\ \sigma_2^*(L)
      \end{array} } 
      \right] u_t

The Appendix of chapter 8 of :cite:`HS2013`  explains why the impulse
response functions in the Wold representation estimated by the
econometrician do not resemble the impulse response functions that
depict the response of consumption and the deficit to innovations to
agents' information.

Technically, :math:`\sigma_2(\beta) = [0\,\,\,0]` implies that the
history of :math:`u_t`\ s spans a *smaller* linear space than does the
history of :math:`w_t`\ s.

This means that :math:`u_t` will typically be a distributed lag of
:math:`w_t` that is not concentrated at zero lag:

.. math::  u_t = \sum_{j=0}^\infty \alpha_j w_{t-j} 

Thus, the econometrician's news :math:`u_t` potentially responds
belatedly to agents' news :math:`w_t`.

We will construct Figures from Chapter 8 Appendix E of :cite:`HS2013` to
illustrate these ideas:

.. code:: ipython3

    # This is Fig 8.E.1 from p.188 of HS2013
    
    Econ1.irf(ts_length=40,shock=None)
    
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(Econ1.c_irf,label='Consumption')
    plt.plot(Econ1.c_irf-Econ1.d_irf[:,0].reshape(40,1),label='Deficit')
    plt.legend(loc='best')
    plt.title('Response to $w_{1t}$')
    
    shock2 = np.array([[0],[1]])
    Econ1.irf(ts_length=40,shock=shock2)
    
    plt.subplot(122)
    plt.plot(Econ1.c_irf,label='Consumption')
    plt.plot(Econ1.c_irf-Econ1.d_irf[:,0].reshape(40,1),label='Deficit')
    plt.legend(loc='best')
    plt.title('Response to $w_{2t}$');




The above figure displays the impulse response of consumption and the
deficit to the endowment innovations.

Consumption displays the characteristic "random walk" response with
respect to each innovation. Each endowment innovation leads to a
temporary surplus followed by a permanent net-of-interest deficit.

The temporary surplus just offsets the permanent deficit in terms of
expected present value.

.. code:: ipython3

    G_HS = np.vstack([Econ1.Sc,Econ1.Sc-Econ1.Sd[0,:].reshape(1,8)])
    H_HS = 1e-8 * np.eye(2) #Set very small so there is no measurement error
    LSS_HS = qe.LinearStateSpace(Econ1.A0, Econ1.C, G_HS, H_HS)

.. code:: ipython3

    HS_kal = qe.Kalman(LSS_HS)
    w_lss = HS_kal.whitener_lss() 
    ma_coefs = HS_kal.stationary_coefficients(50, 'ma')
    
    # This is Fig 8.E.2 from p.189 of HS2013
    
    ma_coefs = ma_coefs
    jj = 50
    y1_w1 = np.empty(jj)
    y2_w1 = np.empty(jj)
    y1_w2 = np.empty(jj)
    y2_w2 = np.empty(jj)
    
    for t in range(jj):
        y1_w1[t] = ma_coefs[t][0, 0]
        y1_w2[t] = ma_coefs[t][0, 1]
        y2_w1[t] = ma_coefs[t][1, 0]
        y2_w2[t] = ma_coefs[t][1, 1]
        
    # This scales the impulse responses to match those in the book
    y1_w1 = sqrt(HS_kal.stationary_innovation_covar()[0,0])*y1_w1
    y2_w1 = sqrt(HS_kal.stationary_innovation_covar()[0,0])*y2_w1
    y1_w2 = sqrt(HS_kal.stationary_innovation_covar()[1,1])*y1_w2
    y2_w2 = sqrt(HS_kal.stationary_innovation_covar()[1,1])*y2_w2
    
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(y1_w1,label='Consumption')
    plt.plot(y2_w1,label='Deficit')
    plt.legend(loc='best')
    plt.title('Response to $u_{1t}$')
    plt.subplot(122)
    plt.plot(y1_w2,label='Consumption')
    plt.plot(y2_w2,label='Deficit')
    plt.legend(loc='best')
    plt.title('Response to $u_{2t}$');



The above figure displays the impulse response of consumption and the
deficit to the innovations in the econometrician's Wold representation.

-  this is the object that would be recovered from a high order vector
   autoregression on the econometrician's observations

Consumption responds only to the first innovation.

-  this is indicative of the Granger causality imposed on the
   :math:`[c_t, c_t - d_t]` process by Hall's model: consumption Granger
   causes :math:`c_t - d_t`, with no reverse causality.

.. code:: ipython3

    # This is Fig 8.E.3 from p.189 of HS2013
    
    jj = 20
    irf_wlss = w_lss.impulse_response(jj)
    ycoefs = irf_wlss[1]
    # Pull out the shocks
    a1_w1 = np.empty(jj)
    a1_w2 = np.empty(jj)
    a2_w1 = np.empty(jj)
    a2_w2 = np.empty(jj)
    
    for t in range(jj):
        a1_w1[t] = ycoefs[t][0, 0]
        a1_w2[t] = ycoefs[t][0, 1]
        a2_w1[t] = ycoefs[t][1, 0]
        a2_w2[t] = ycoefs[t][1, 1]
    
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(a1_w1,label='Consumption innov.')
    plt.plot(a2_w1,label='Deficit innov.')
    plt.title('Response to $w_{1t}$')
    plt.legend(loc='best')
    plt.subplot(122)
    plt.plot(a1_w2,label='Consumption innov.')
    plt.plot(a2_w2,label='Deficit innov.')
    plt.legend(loc='best')
    plt.title('Response to $w_{2t}$');


The above figure displays the impulse responses of :math:`u_t` to
:math:`w_t`, as depicted in:

.. math::  u_t = \sum_{j=0}^\infty \alpha_j w_{t-j} 

While the responses of the innovations to consumption are concentrated
at lag zero for both components of :math:`w_t`, the responses of the
innovations to :math:`(c_t - d_t)` are spread over time (especially in
response to :math:`w_{1t}`).

Thus, the innovations to :math:`(c_t - d_t)` as revealed by the vector
autoregression depend on what the economic agent views as "old news"
