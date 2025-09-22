#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python3
"""
MLE & Distributions: A Hands-on Script with Plots
-------------------------------------------------
Run this as a script (python mle_distributions_demo.py) or inside Jupyter:
    %run mle_distributions_demo.py

This script explains and demonstrates Maximum Likelihood Estimation (MLE)
using Bernoulli, Normal, and Gamma distributions.

Each section:
1. Simulates data from the distribution (with known "true" parameters).
2. Estimates parameters using MLE.
3. Plots the likelihood or fitted probability density function (PDF).
4. Provides detailed comments for learning.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import norm, gamma as gamma_dist

# Always fix the random seed for reproducibility
np.random.seed(42)


# -----------------------------
# Helper to print explanations
# -----------------------------
def explain(title, text):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))
    print(text)


# -----------------------------
# 0) Quick refresher on MLE
# -----------------------------
def intro_mle():
    explain(
        "What is MLE (Maximum Likelihood Estimation)?",
        """MLE chooses parameter estimates that maximize the likelihood function.
        
Example:
- Suppose you observe coin flips (data).
- Likelihood is the probability of observing exactly this data as a function of p (probability of heads).
- MLE picks p-hat that makes your observed data the most probable.

We’ll apply this to:
- Bernoulli (coin flip style, binary data).
- Normal (bell-shaped continuous data).
- Gamma (positive, skewed data).
"""
    )


# -----------------------------
# 1) Bernoulli Distribution
# -----------------------------
def bernoulli_demo(true_p=0.7, n=100):
    """
    Bernoulli distribution models binary outcomes:
    - 1 = success (probability = p)
    - 0 = failure (probability = 1-p)

    Variables:
    - true_p : the actual (unknown in real life) probability of success
    - n      : number of independent trials (sample size)
    """
    explain(
        "Bernoulli Distribution",
        f"""For k successes in n trials:
Likelihood(p) = p^k (1-p)^(n-k)

MLE estimate:
p-hat = k / n (sample mean)

Simulation settings: true p={true_p}, n={n}"""
    )

    # Generate sample: each trial is 1 if random < true_p, else 0
    sample = (np.random.rand(n) < true_p).astype(int)

    # k = number of successes (sum of 1s)
    k = sample.sum()

    # p-hat = sample mean = proportion of successes
    phat = sample.mean()

    print(f"Observed successes: {k}/{n} -> p-hat (MLE) = {phat:.4f}")

    # Compute log-likelihood curve for a grid of p values
    p_grid = np.linspace(1e-4, 1-1e-4, 400)
    loglike = k*np.log(p_grid) + (n-k)*np.log(1 - p_grid)

    # Plot log-likelihood
    plt.figure()
    plt.plot(p_grid, loglike)
    plt.axvline(phat, linestyle="--", label="MLE p-hat")
    plt.title("Bernoulli: Log-Likelihood vs p")
    plt.xlabel("p")
    plt.ylabel("log L(p)")
    plt.legend()
    plt.show()


# -----------------------------
# 2) Normal Distribution
# -----------------------------
def normal_demo(true_mu=1.0, true_sigma=0.5, n=200):
    """
    Normal distribution: X ~ N(mu, sigma^2)
    Variables:
    - true_mu    : true mean of the distribution
    - true_sigma : true standard deviation
    - n          : number of observations (sample size)
    """
    explain(
        "Normal Distribution",
        f"""MLE estimates:
μ-hat = sample mean
σ-hat^2 = (1/n)*Σ (x_i - μ-hat)^2  (MLE variance)

Simulation: true μ={true_mu}, true σ={true_sigma}, n={n}"""
    )

    # Generate sample data from Normal(true_mu, true_sigma)
    x = np.random.normal(true_mu, true_sigma, n)

    # MLE for mu = sample mean
    mu_hat = x.mean()

    # MLE for sigma = sqrt of variance with denominator n
    sigma_hat = np.sqrt(np.mean((x - mu_hat)**2))

    print(f"Estimated μ-hat = {mu_hat:.4f}, σ-hat (MLE) = {sigma_hat:.4f}")

    # Histogram of sample with fitted normal PDF overlay
    grid = np.linspace(x.min(), x.max(), 400)
    fitted_pdf = norm.pdf(grid, loc=mu_hat, scale=sigma_hat)

    plt.figure()
    plt.hist(x, bins=30, density=True, alpha=0.5, label="Sample data")
    plt.plot(grid, fitted_pdf, linewidth=2, label="Fitted Normal (MLE)")
    plt.title("Normal: Histogram + Fitted PDF (MLE)")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


# -----------------------------
# 3) Gamma Distribution
# -----------------------------
def gamma_demo(true_alpha=2.0, true_beta=3.0, n=300):
    """
    Gamma distribution (parameterization used here):
    - shape (alpha > 0)
    - rate (beta > 0)
    Note: scipy uses scale = 1/beta internally.

    Variables:
    - true_alpha : true shape
    - true_beta  : true rate
    - n          : sample size
    """
    explain(
        "Gamma Distribution",
        f"""MLE approach:
Profile likelihood method:
1. Given alpha, beta-hat(alpha) = alpha / sample mean
2. Substitute into log-likelihood, maximize wrt alpha
3. Compute beta-hat

Simulation: true α={true_alpha}, β={true_beta}, n={n}"""
    )

    # Simulate gamma sample (SciPy expects scale = 1/beta)
    sample = np.random.gamma(shape=true_alpha, scale=1.0/true_beta, size=n)

    # Sample mean (used in profile likelihood)
    xbar = sample.mean()

    # Define profile log-likelihood function for alpha
    def loglike_alpha(alpha):
        beta_hat = alpha / xbar
        return (n*(alpha*np.log(beta_hat) - math.lgamma(alpha))
                + (alpha-1)*np.sum(np.log(sample))
                - beta_hat*np.sum(sample))

    # Numerically maximize log-likelihood wrt alpha
    res = optimize.minimize_scalar(lambda a: -loglike_alpha(a),
                                   bounds=(1e-3, 50), method="bounded")

    alpha_hat = res.x
    beta_hat = alpha_hat / xbar

    print(f"Estimated α-hat = {alpha_hat:.4f}, β-hat = {beta_hat:.4f}")

    # Plot histogram of data + fitted gamma PDF
    grid = np.linspace(sample.min(), sample.max(), 400)
    fitted_pdf = gamma_dist.pdf(grid, a=alpha_hat, scale=1.0/beta_hat)

    plt.figure()
    plt.hist(sample, bins=30, density=True, alpha=0.5, label="Sample data")
    plt.plot(grid, fitted_pdf, linewidth=2, label="Fitted Gamma (MLE)")
    plt.title("Gamma: Histogram + Fitted PDF (MLE)")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    intro_mle()
    bernoulli_demo()
    normal_demo()
    gamma_demo()


# In[ ]:





# In[ ]:




