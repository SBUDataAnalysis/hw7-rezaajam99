#!/usr/bin/env python
# coding: utf-8

# # Q1
# 
# Consider the probability distribution below:
# 
# $$P(x) = e^{0.4(x-0.4)^2 - 0.08 x^4}$$
# 

# ## (a)
# Write a function to implement Metropolice algorithm to generate random samples that have the same PDF as the given formula.

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from random import random


# In[68]:


x = np.linspace(-5, 5, 100)
def P(x):
    return np.exp(0.4*((x-0.4)**2)-0.08*x**4)
plt.plot(x, P(x), color='red')
plt.show()


# In[61]:


def Metropolis(num, posterier, std):
    first = np.random.normal(0, std)
    sample = [first]
    for i in range(num):
        next_step = np.random.normal(first, std)
        alpha = posterier(next_step) / posterier(first)
        if np.random.uniform() <= alpha:
            sample.append(next_step)
            first = next_step
    return np.array(sample)


# ## (b)
# Plot the function alongside with the histogram of the generated sample. Do they match? If not, what shall be done here?

# In[115]:


bins = 100 # the number of block


# In[116]:


std=float(input(" Enter the standard deviation: "))
plt.hist(Metropolis(n, P, std), bins, density = True, label = 'Prob',color='purple')
X = np.linspace(-4, 4, bins)
vector_P = np.vectorize(P)
Y = vector_P(X)
Y = Y / np.sum(Y) * bins / 8
plt.plot(X, Y, label = 'Normalized function')
plt.legend()
plt.show()


# ## (c)
# Generate data using Metropolice algorithm with four different values for step size (0.1, 0.5, 1 and 2) and plot the autocorrelation function for each case. which one becomes uncorrelated sooner?

# In[51]:


from statsmodels.graphics import tsaplots


# In[56]:


for i in [0.1,0.5,1,2]:
    fig = tsaplots.plot_acf(np.array(Metropolis(n, P, i)), lags=10)
    plt.show()


# # Q2: Variational Monte Carlo

# Imagine we have a quantum harmonic osscilator were the Hamiltonian is
# $$\hat{H} = -\frac{1}{2}\frac{d^2}{dx^2} + \frac{1}{2}x^2 $$
# 
# In oreder to solve this, we guess a wave function, and then using Monte Carlo methods, we try to find the lowest energy state.
# 
# Now imagine our wave function is
# 
# $$ \psi \propto e^{-\lambda x^2} $$
# 
# Then the local energy of the system at each point would be
# 
# $$ E_L = \frac{H \psi}{\psi} = \lambda + x^2 (\frac{1}{2} - 2\lambda^2) $$

# ## (a)
# 
# Write a Metropolice algorithm to generate values for $x$ as a function of $\lambda$. Then plot the PDF alongside with the analitycal function for $P(x) = \psi^2(x)$ to confirm that they match.
# 
# P.S. in quantum mechanics, the probability function is always given as $\psi^2$.

# In[113]:


def psi(x):
    return np.exp(-2 * landa * x**2)


# In[122]:


x=np.linspace(2.0, 3.0, num=5)
bins=100
for i in x :
    landa=float(input(" Enter the amout of wavelength: "))    
    s=float(input(" Enter the standard deviation: "))
    plt.hist(Metropolis(n, psi, s), bins, density = True, label = 'Prob',color='purple')
    r = np.linspace(np.min(psi), np.max(psi), bins)
    vector_P = np.vectorize(psi)
    g = vector_P(r)
    g = Y / np.sum(Y) * bins / 8
    plt.plot(r, g, label = 'Normalized function')
    plt.plot(x, psi, color='green')
    plt.legend()
    plt.show()    


# ## (b)
# 
# Write two functions to calculate the expectation value $\langle E_L \rangle$ and $\sigma^2 = \langle E_L^2 \rangle - \langle E_L \rangle^2$ as a function of $\lambda$.

# In[ ]:


#Your code here


# ## (c)
# 
# Generate a uniform sample of values for $\lambda$, plot the mean and the variance and find the best value for $\lambda$ (The minimum energy value).

# In[31]:


...


# ## (d)
# 
# What's the sinificance of $\sigma^2$? Explain in a few lines in Farsi or English.

# 

# # Importance sampling

# A frequently used method is Importance Sampling. In this method, a proxy distribution is introduced to sample random numbers from any distribution. In most cases, we choose a well-known distribution such as Gaussian distribution, uniform distribution as a proxy distribution. The main concept of this method can be simply written in the following form, where q(x) is a proxy distribution.
# 

# ![1_nwk7MyNdeFgACxQ832oq8Q.png](attachment:1_nwk7MyNdeFgACxQ832oq8Q.png)

# With Importance Sampling, instead of generating random numbers from p(x), we choose i.i.d samples ${x_i} (i=1,2,…,n)$ from a proxy distribution q(x) and approximate the integration value with the following calculation. Here, p(x)/q(x) is called importance of the sampling.

# ![1_BwRHc3Njf0FbIjhNpYulEg.png](attachment:1_BwRHc3Njf0FbIjhNpYulEg.png)

# Now, let’s use a Laplace distribution’ variance calculation as an example.
# Consider f(x)=x² and a probability density function p(x)=1/2 Exp(-|x|). The distribution with density function like p(x) is called Laplace distribution.
# If we choose a uniform distribution as a proxy distribution, the variance of the Laplace distribution can be approximately calculated by

# ![1_JRuYsXh2E5bmPwwa7FzjBg.png](attachment:1_JRuYsXh2E5bmPwwa7FzjBg.png)

# ![1_J5Us-CZxnvHL-Dd1xzpxBw.png](attachment:1_J5Us-CZxnvHL-Dd1xzpxBw.png)

# With paper and a pencil, we can easily calculate the Var[x]. The value of this calculation is 2. Now confirm the result of the Importance Sampling method.

# # Inverse Transform Sampling

# In Inverse Transform Sampling method, we use a random number u generated from a 1-dimensional uniform distribution to generate a random number x of any 1-dimensional probability density function p(x). In this case, we use the inverse function of the cumulative distribution function of p(x). If the cumulative distribution function of p(x) is P(x), then the inverse function of u=P(x) is x =P^-1 (u). Now, $x = P^-1 (u)$ has p(x) as probability density function on [0,1]. Therefore, with n samples from uniform distribution on [0,1] ${u_i} (i=1,2,…,n)$, we can generate n sample of p(x) distribution ${x_i} (i=1,2,…,n)$ by calculate $x_i = P^{-1}(u_i)$.
# 
# Again, let's consider the Laplace distribution’s variance calculation as an example. This time we directly generate random numbers (samples) from the Laplace distribution’s probability density function using the Inverse Transform Sampling method. With these random numbers, we will again recalculate the approximated value of Var[x].

# ![1_OeecUj7UVwBosCoQSW8GnQ.png](attachment:1_OeecUj7UVwBosCoQSW8GnQ.png)

# Now check the result of this method.

# In[105]:


numbers = [10*2,10*3,10*4,10*5, 10*6, 10*7, 10*8]
variance = []
for i in numbers:
    uni_f = np.random.uniform(0, 1, i)
    sample = - np.sign(uni_f - 1 / 2) * np.log(1 - 2 * np.abs(uni_f - 1 / 2))
    variance.append(np.mean(sample ** 2))
plt.semilogx(numbers, variance, color='g')


# # Rejection Sampling 

# ![1_CmlQbGdECPyfM5vZeEkHUg.png](attachment:1_CmlQbGdECPyfM5vZeEkHUg.png)

# The idea in Rejection Sampling is to use a proxy distribution (Gaussian or uniform distribution, etc.) called q(x) to generate a random number and use another uniform distribution to evaluate the generated sample whether or not to accept it as a sample generated from p(x). With this method, we can also generate random numbers from a higher dimensional distribution.
# 
# As preparation in generating random numbers with this method, we need to know a finite value of L where max[p(x)/q(x)] < L. Here, q(x) is a proxy distribution.
# 
# First, we generate a random number x’ from a proxy distribution q(x). This x’ is called a proposal point.<br>
# Next, generate a random number v from a uniform distribution on [0, L]. This v will be used to evaluate the proposal point, whether to be fine considering generated from p(x).<br>
# If v ≤ p(x’)/q(x’), then x’ is accepted as a random number generated by p(x), else, x’ is rejected.
# The algorithm in generating n random numbers with Rejection Sampling is

# ![1_ZkD77xvkprPSVe36-ao9SQ.png](attachment:1_ZkD77xvkprPSVe36-ao9SQ.png)

# Now use rejection sampling method to evaluate the variance of Laplace Distribution.

# In[ ]:


#code here


# # Markov Chain Monte carlo method(MCMC)

# In the Rejection sampling method, it is impossible to generate random numbers when the upper boundary L is not known. MCMC method is an effective solution to this problem. MCMC method uses the concept of a stochastic process (Markov chain in this case). In this case, the generation of i-th sample $x_i$ depends on the previous sample $x_{i-1}$. ${x_1, x_2, …,x_n}$ with this concept is called a Markov chain. Here, I introduce one of the MCMC methods, the Metropolis-Hastings method.<br>
# 
# The process in this method is similar to Rejection sampling. But here, a proxy distribution density function is represented by a conditional probability $q(x|x_i)$, and the evaluation index v is generated from a uniform distribution on [0,1].<br>
# 
# First, we generate a random number x’ from a proxy distribution $q(x|x_i)$. This x’ is called a proposal point.<br>
# Next, generate a random number v from a uniform distribution on [0, 1]. This v will be used to evaluate the proposal point, whether to be fine considering generated from p(x).<br>
# If $v ≤ p(x’)q(x_i|x’)/(p(x_i)q(x’|x_i))$, then x’ is accepted as a random number generated by p(x), else, x’ is rejected.<br>
# The algorithm in generating n random numbers with Rejection Sampling is

# ![1_smfr3QVGxDzcGUFUKFqvhQ.png](attachment:1_smfr3QVGxDzcGUFUKFqvhQ.png)

# Now use MCMC to calculate the variance of Laplace Distribution.

# In[2]:


#code here

