from scipy.stats import gumbel_r
import copy
import numpy as np

class CD(object):
    # short for Contrastive Divergence
    def __init__(self, scen):
        """
        Initiate an CD objected
        Only implemented for Gumbel randomization.

        scen: specify the data dist.
        """
        if (not scen) or (scen[0].get_distr() != "normal"):
            raise NotImplementedError("Only implemented for normal dist.")

        self.scen = scen
        self.num_actions = len(scen)
        self.choices = [] # history of arm choices
        self.rewards= []
        self.scales = [] # list of added randomization noise scales

    def update(self, action, reward, rand_scale, extra=None):
        """
        Bookkeeping.
        
        extra: algorithm dependent extra info. 
        UCB algs: extra is the ucb_bounds
        thompson sampling: expected reward
        """
        self.choices.append(action)
        self.rewards.append(reward)
        self.scales.append(rand_scale)


    def setup_sampling(self, params=None):
        """
        set up the initial sampling framework, 
        turn all the lists into matrices

        compute the log-likelihood for the softmax
        and its corresponding gradient

        X: is the arms average up to time T
        S: is the decision statistic that we use to inform decisions
        hmu: is the estimated mean (initial)
        """
        self.X, self.num_pulls = payoff_matrix(self.num_actions, self.choices, self.rewards) 
        self.S = self.get_decision_stat()
        #print "choices:", self.choices
        #print "The number of pulls:", self.num_pulls
        if params:
            self.hmu = params
        else:
            # initialize the mean estimate to be the empirical mean
            self.hmu = copy.deepcopy(self.X[-1,:])

    def get_decision_stat(self):
        """
        compute the decision statistics from the data,
        specific to the bandit algorithm
        """
        raise NotImplementedError("Abstract method")


    def sampler(self, cycles=5, mcmc_stepsize=0.5, tune_interval=15):
        """
        Metropolis-Hastings sampler for the specified densities.
        """

        # initialize the sampler.
        self.init_sampler()

        self.approx_grad = np.zeros(self.num_actions)
        burnin = cycles / 2
        for i in range(cycles):
            self.total += 1
            new, ll_new, ll_S_new, grad, ll_pos_new = self.proposal(mcmc_stepsize)
            log_ratio = ll_new + ll_S_new + ll_pos_new - self.ll - self.ll_S - self.ll_pos
            #print ll_pos_new, self.ll_pos
            #print ll_new, self.ll
            #print ll_S_new, self.ll_S

            # Metropolis-Hastings accept/reject
            if np.log(np.random.uniform()) < log_ratio:
                self.set_state(new, grad)
                self.ll, self.ll_S, self.ll_pos = ll_new, ll_S_new, ll_pos_new
                self.accept += 1

            if i > burnin:
                self.approx_grad += self.state_grad / (cycles - burnin)

    def tune(self, mcmc_stepsize, acc_rate):
        if acc_rate < 0.001:
            # reduce by 90 percent
            return mcmc_stepsize * 0.1
        elif acc_rate < 0.05:
            # reduce by 50 percent
            return mcmc_stepsize * 0.5
        elif acc_rate < 0.2:
            # reduce by ten percent
            return mcmc_stepsize * 0.9
        elif acc_rate > 0.95:
            # increase by factor of ten
            return mcmc_stepsize * 10 
        elif acc_rate > 0.75:
            # increase by double
            return mcmc_stepsize * 2 
        elif acc_rate > 0.5:
            # increase by ten percent
            return mcmc_stepsize * 1.1 
        else:
            return mcmc_stepsize

    def next(self, sgd_stepsize=0.1, mcmc_stepsize=0.5, cycles=50):

        self.sampler(cycles=cycles, mcmc_stepsize=mcmc_stepsize) # compute the approximate gradient at the current hmu

        self.hmu += sgd_stepsize * (self.data_grad - self.approx_grad)
        accept_ratio = self.accept * 1.0 / self.total

        return self.hmu, accept_ratio
    
    def proposal(self, mcmc_stepsize=0.5):
        """
        Samples from the proper proposal distribution,
        and the corresponding transition likelihoods and gradient.

        The proper proposal distribution depends on the bandit alg.
        """
        raise NotImplementedError("Abstract method")
    

    def softmax(self, S):

        # if no randomization return 0
        if (np.array(self.scales[self.num_actions:]) == 0).any():
            return 0

        ll_S = 0
      
        for t, action in enumerate(self.choices):
            if t >= self.num_actions:
                ll_S += S[t, action] / self.scales[t] - \
                        np.log(np.exp(S[t, :] / self.scales[t]).sum())

        return ll_S

    def log_likelihood(self, reward, choices, params):
        """
        Gaussian likelihood.
        TODO: make this an abstract method and implement with
        specific distributions
        """
        num_actions = len(self.scen)

        vec = np.array(reward)
        variance = np.array([self.scen[i].get_sigma()**2 for i in range(num_actions)])

        ll = - ((vec - params[choices])**2 / (2 * variance[choices])).sum()
        return ll


    def log_likelihood_gradient(self, x, params):
        """
        gradient of Gaussian log likelihood. 
        TODO: same with log likelihood
        """
        num_actions = len(self.scen)

        variance = np.array([self.scen[i].get_sigma()**2 * 1.0 / self.num_pulls[i]
            for i in range(num_actions)])

        grad = -(params - x) / variance 
        return grad


def payoff_matrix(num_actions, choices, reward):
    if len(choices) != len(reward):
        raise ValueError("Input choice length not equal to reward length")
    
    T = len(choices) # episode T
    X = np.zeros((T, num_actions)) # the matrix of arm means
    payoff_sums = np.zeros(num_actions)
    num_pulls = np.zeros(num_actions)

    # initialization
    for t, action in enumerate(choices): 
        payoff_sums[action] += reward[t]
        num_pulls[action] += 1
        if t < num_actions:
            X[t, :(t+1)] = payoff_sums[:(t+1)] * 1.0 / num_pulls[:(t+1)]
        else:
            X[t, :] = payoff_sums * 1.0 / num_pulls

    return X, num_pulls
