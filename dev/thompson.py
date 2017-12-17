import numpy as np
import copy
from debias import CD, payoff_matrix

# thompson_pf: same thing as thompson sampling, except we pull every arm once first.
def thompson_pf(scen, rand=None, gumbel_t_varies=False, held=False): 
    num_actions = len(scen)
    payoff_sums = np.zeros(num_actions)
    payoff_sums_held = np.zeros(num_actions)
    num_pulls = np.zeros(num_actions, dtype=np.int8)
    num_pulls_held = np.zeros(num_actions, dtype=np.int8)
    rand_exp_reward = np.zeros(num_actions)
    debiaser = thompson_sgd(scen)
    t = 0

    # initialize empirical sums
    for i in range(num_actions):
        reward = scen[i].get_reward()
        scen[i].update_posterior(reward)
        num_pulls[i] += 1
        num_pulls_held[i] += 1
        payoff_sums[i] = reward
        payoff_sums_held[i] = reward
        ind_sample = True
        debiaser.update(i, reward, 0, np.zeros(num_actions))
        yield i, reward, num_pulls, payoff_sums, None, debiaser, num_pulls_held, payoff_sums_held, ind_sample

    t = num_actions
    last_action = None

    while True:
        if held and last_action is not None:
            action = last_action
            last_action = None
            ind_sample = True
        else:
            ind_sample = False
            decision_stat = np.array([scen[i].expected_reward_drawn_from_belief() for i in range(num_actions)])
            rand_exp_reward = decision_stat.copy()
            if rand:
                if gumbel_t_varies:
                    rand_exp_reward += rand.rvs(size=num_actions) / np.sqrt(t)
                else:
                    rand_exp_reward += rand.rvs(size=num_actions)
            action = np.argmax(rand_exp_reward)
            last_action = action
        reward = scen[action].get_reward()
        if not held or (held and not ind_sample):
            num_pulls[action] += 1
            payoff_sums[action] += reward 
            scen[action].update_posterior(reward)
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
        debiaser.update(action, reward, scale, decision_stat)

        yield action, reward, num_pulls, payoff_sums, None, debiaser, num_pulls_held, payoff_sums_held, ind_sample
        t = t + 1


class thompson_sgd(CD):
    """
    This is a subclass of CD that implements the
    SGC for Thompson's sampling
    """

    def __init__(self, scen):
        CD.__init__(self, scen)

        # Record the expected reward drawn at every round
        self.expected_reward = []
        self.prior_mu = np.array([s.get_param()[0] for s in scen])
        self.prior_sigma = np.array([s.get_param()[1] for s in scen])

    def update(self, action, reward, rand_scale, extra):
        """
        extra: samples from the posterior dist.
        """
        CD.update(self, action, reward, rand_scale)

        self.expected_reward.append(extra)

    def get_decision_stat(self):
        return np.array(self.expected_reward)

    def set_state(self, prev_state, grad):
        self.state = list(prev_state)
        self.state_grad = copy.deepcopy(grad)

    def init_sampler(self):
        self.total, self.accept = 0, 0

        self.ll = self.log_likelihood(self.rewards, self.choices, self.hmu)
        self.ll_S = 0 #self.softmax(self.S)
        # gradient evaluated on the data at last iteratio of hmu 
        self.data_grad = self.log_likelihood_gradient(self.X[-1,:], self.hmu)
        self.ll_pos = self.posterior_log_likelihood(self.X, self.S)

        # initialize the state to the data
        self.set_state(self.rewards, self.data_grad)

    def proposal(self, mcmc_stepsize=0.5):
        """
        The proposal distribution used for the reward is to add
        normal distribution with the known variances.
        """
        new = [self.state[t] + mcmc_stepsize * self.scen[a].get_sigma() * np.random.standard_normal()\
                for t, a in enumerate(self.choices)]

        X_new, _ = payoff_matrix(self.num_actions, self.choices, new)
        ll_new = self.log_likelihood(new, self.choices, self.hmu)
        ll_S_new = 0 #self.softmax(S_new)
        grad = self.log_likelihood_gradient(X_new[-1,:], self.hmu)
        ll_pos = self.posterior_log_likelihood(X_new, self.S)
        #print S_new[5:,:].argmax(1)
        #print self.S[5:,:].argmax(1)
        #print S_new
        #print self.S

        return new, ll_new, ll_S_new, grad, ll_pos

    def posterior_log_likelihood(self, X, S):
        """
        Compute posterior likelihood for any two matrices, X and S
        """
        start = self.num_actions 

        sigma = np.array([s.get_sigma() for s in self.scen]) # data sigmas for different arms
        num_pulls = np.ones(self.num_actions) # start with pulling each arm once

        ll = 0

        for t, action in enumerate(self.choices):
            # compute posterior mean and sigma based on PREVIOUS observations
            if t >= start:
                pmu = (self.prior_sigma ** 2 * X[(t-1),:] + sigma ** 2 * self.prior_mu / num_pulls) / \
                        (self.prior_sigma ** 2 + sigma ** 2 / num_pulls)
                psigma = (self.prior_sigma ** 2 * sigma ** 2 / \
                        (sigma ** 2 + num_pulls * self.prior_sigma ** 2)) ** 0.5
                #S[t,:] = np.random.normal(pmu, psigma)
                ll += - ((S[t,:] - pmu)**2 / (2 * psigma**2)).sum()

                num_pulls[action] += 1
                #print pmu
                #print psigma

        return ll

