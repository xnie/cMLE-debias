import math
import pdb
import random
import copy
import numpy as np
from debias import CD, payoff_matrix

def ucb(scen, greedy=False, lil_ucb=False, egreedy=False, rand=None, alpha=9, beta=1, lil_epsilon=0.01, greedy_epsilon=0.05, delta=0.005, gumbel_t_varies=False, held=False):
    # scen: a list of reward distr
    # rand: randomization noise generator 
    num_actions = len(scen)

    payoff_sums = np.zeros(num_actions)
    payoff_sums_held = np.zeros(num_actions)

    num_pulls = np.zeros(num_actions, dtype=np.int8)
    num_pulls_held = np.zeros(num_actions, dtype=np.int8)

    ucbs = np.zeros(num_actions)
    debiaser = ucb_sgd(scen)
    if egreedy:
        debiaser.set_eps(greedy_epsilon)

    # initialize empirical sums
    for t in range(num_actions):
       payoff_sums[t] = scen[t].get_reward()
       payoff_sums_held[t] = payoff_sums[t]
       num_pulls[t] += 1
       num_pulls_held[t] += 1
       debiaser.update(t, payoff_sums[t], 0, ucbs)
       ind_sample = True
       yield t, payoff_sums[t], num_pulls, payoff_sums, ucbs, debiaser, num_pulls_held, payoff_sums_held, ind_sample 

    t = num_actions
    last_action = None

    while True:
        if held and last_action is not None:
            action = last_action
            ind_sample = True
            last_action = None
        else:
            ind_sample = False
            if greedy or egreedy:
                ucbs_bound = np.zeros(num_actions)
            else:
                ucbs_bound = np.array([scen[i].get_upper_bound(t, num_pulls[i], lil_ucb, alpha, beta, lil_epsilon, delta) for i in range(num_actions)])
            ucbs = np.array([payoff_sums[i] / num_pulls[i] + ucbs_bound[i] for i in range(num_actions)])
            if rand:
                if gumbel_t_varies:
                    ucbs += rand.rvs(size=num_actions) / np.sqrt(t)
                else:
                    ucbs += rand.rvs(size=num_actions)
            mx = max(ucbs)
            all_maxes = [i for i in range(num_actions) if ucbs[i] == mx]
            if egreedy:
                if random.random() < greedy_epsilon:
                    action = random.choice(range(num_actions))
                else:
                    action = random.choice(all_maxes)
            else:
                action = random.choice(all_maxes)
            last_action = action

        reward = scen[action].get_reward()
        if not held or (held and not ind_sample):
            num_pulls[action] += 1
            payoff_sums[action] += reward 
        else:
            num_pulls_held[action] += 1
            payoff_sums_held[action] += reward 
        if rand:
            if gumbel_t_varies:
                scale = rand.kwds["scale"] / np.sqrt(t)
            else:
                scale = rand.kwds["scale"] 
        else:
            scale = 0 
        debiaser.update(action, reward, scale, ucbs_bound)

        yield action, reward, num_pulls, payoff_sums, ucbs_bound, debiaser, num_pulls_held, payoff_sums_held, ind_sample
        t = t + 1


class ucb_sgd(CD):
    """
    This is a subclass of CD that implements the 
    SGD for UCB type bandit algorithms
    """

    def __init__(self, scen):
        CD.__init__(self, scen) 

        # The UCB bounds are specific to UCB algorithms
        self.ucb_bounds = []

    def set_eps(self, epsilon):
        self._eps = epsilon

    def update(self, action, reward, rand_scale, extra):
        """
        extra: ucb_bounds
        """
        CD.update(self, action, reward, rand_scale)

        self.ucb_bounds.append(extra)

    def get_decision_stat(self):
        """
        add the ucb bounds to the arm means.

        X: is the arms averages up to time T
        """
        if not hasattr(self, "X"):
            raise ValueError("Data matrix self.X is not computed.")

        self.ucb_upward = np.array(self.ucb_bounds)
        S = self.decision_stat(self.X)
        return S 

    def proposal(self, mcmc_stepsize=0.5):
        """
        The proposal distribution used for the reward is to add
        normal distribution with the known variances.
        """
        new = [self.state[t] + mcmc_stepsize * self.scen[a].get_sigma() * np.random.standard_normal()\
                for t, a in enumerate(self.choices)]

        X_new, _ = payoff_matrix(self.num_actions, self.choices, new)
        S_new = self.decision_stat(X_new)
        ll_new = self.log_likelihood(new, self.choices, self.hmu)
        if hasattr(self, "_eps"):
            ll_S_new = self.eps_log_likelihood(S_new) 
        else:
            ll_S_new = self.softmax(S_new)
        grad = self.log_likelihood_gradient(X_new[-1,:], self.hmu)

        return new, ll_new, ll_S_new, grad, 0

    def set_state(self, reward, grad):
        self.state = copy.deepcopy(list(reward))
        self.state_grad = copy.deepcopy(grad)

    def init_sampler(self):
        self.total, self.accept = 0, 0

        self.ll = self.log_likelihood(self.rewards, self.choices, self.hmu)
        if hasattr(self, "_eps"):
            self.ll_S = self.eps_log_likelihood(self.S) 
        else:
            self.ll_S = self.softmax(self.S)
        # gradient evaluated on the data at last iteratio of hmu 
        self.data_grad = self.log_likelihood_gradient(self.X[-1,:], self.hmu)
        self.ll_pos = 0

        # initialize the state to the data
        self.set_state(self.rewards, self.data_grad)

    def decision_stat(self, X):
        if not hasattr(self, "ucb_upward"):
            raise ValueError("self.ucb_upward does not exist!")

        S = np.zeros(X.shape)
        T = S.shape[0] # total rounds
        start = S.shape[1] # time to start making decisions based on data 

        for t in range(start, T):
            # the decision statistic at round t is the sums of
            # PREVIOUS means and UCB bounds at current time.
            S[t,:] = X[(t-1),:] + self.ucb_upward[t,:]

        return S

    def eps_log_likelihood(self, S):

        ll_S = 0
        for t, action in enumerate(self.choices):
            if t >= self.num_actions:
                if self.scales[t] == 0:
                    ll_S += np.log((1-self._eps) + self._eps / self.num_actions) if action == np.argmax(S[t,:]) else np.log(self._eps / self.num_actions)
                else:
                    ll_S += np.log((1-self._eps) * np.exp(S[t, action] / self.scales[t]) / np.exp(S[t, :] / self.scales[t]).sum() +
                            self._eps / self.num_actions)

        return ll_S

