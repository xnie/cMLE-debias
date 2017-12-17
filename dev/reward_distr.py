import numpy as np
import pdb
import random
import math

class Gaussian():
    def __init__(self, mu, sigma, mu0, sigma0):
        self.mu = mu 
        self.sigma = sigma
        self.mu0 = mu0
        self.sigma0 = sigma0

    def get_param(self):
        return np.array([self.mu0, self.sigma0])

    def get_distr(self):
        return "normal"

    def get_sigma(self):
        return self.sigma

    def get_reward(self):
        if self.sigma == 0:
            return self.mu
        return np.random.normal(self.mu, self.sigma, 1)[0]

    def get_expected_reward(self, precision=None):
        if precision == None:
            return self.mu
        else:
            return round(self.mu, precision)

    def get_upper_bound(self, t, num_pulls, lil_ucb=False, alpha=9, beta=1, epsilon=0.01, delta=0.005):
        if lil_ucb:
            try:
                return (1+beta)*(1+epsilon**0.5)*(2*(1+epsilon)*math.log(math.log((1+epsilon)*num_pulls)/delta)/num_pulls)**0.5
            except:
                pdb.set_trace()
        else:
            return (8 * math.log(t + 1) * (self.sigma**2) / num_pulls) ** 0.5 

    def expected_reward_drawn_from_belief(self):
        if self.sigma0 == 0:
            return self.mu0
        return np.random.normal(self.mu0, self.sigma0)

    def update_posterior(self, reward):
        if self.sigma == 0:
            self.mu0 = reward
            self.sigma0 = 0
        else:
            self.mu0 = 1.0 * (self.sigma0 ** 2 * reward + self.sigma ** 2 * self.mu0) / (self.sigma0**2 + self.sigma**2) 
            self.sigma0 = ((1.0 * self.sigma ** 2 * self.sigma0 ** 2) / (self.sigma ** 2 + self.sigma0 ** 2)) ** 0.5 #update sigma0

    def get_prior_distr(self):
        return "normal"

    def __str__(self):
        return "gauss_mu_" + str(self.mu) + "_sigma_" + str(self.sigma) + "_mu0_" + str(self.mu0) + "_sigma0_" + str(self.sigma0)

    def __repr__(self):
        return self.__str__()

class Bernoulli():
    def __init__(self, mu, alpha, beta):
        self.mu = mu 
        self.alpha = alpha
        self.beta = beta
        self.sigma0 = 0

    def get_sigma(self):
        return (self.mu * (1-self.mu))**0.5

    def get_param(self):
        return np.array([self.alpha, self.beta])

    def get_distr(self):
        return "ber"

    def get_reward(self):
        return 1 if random.random() < self.mu else 0

    def get_upper_bound(self, t, num_pulls, lil_ucb=False, alpha=9, beta=1, epsilon=0.01, delta=0.005):
        if lil_ucb:
            a = (1+beta)*(1+epsilon**0.5)*(2*(1+epsilon)*math.log(math.log((1+epsilon)*num_pulls)/delta)/num_pulls)**0.5
            return a 
        else:
            return (2 * math.log(t + 1) / num_pulls) ** 0.5 

    def get_expected_reward(self, precision=None):
        if precision == None:
            return self.mu
        else:
            return round(self.mu, precision)

    def expected_reward_drawn_from_belief(self):
        return np.random.beta(self.alpha, self.beta)

    def update_posterior(self, reward):
        if reward > 0:
            self.alpha += 1
        else: 
            self.beta += 1

    def get_prior_distr(self):
        return "beta"

    def __str__(self):
        return "bern_mu_" + str(self.mu) + "_alpha_" + str(self.alpha) + "_beta_" + str(self.beta)

    def __repr__(self):
        return self.__str__()
