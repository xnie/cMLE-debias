import math
import errno
import inspect
from scipy.stats import gumbel_r
import copy
import os
import pdb
import random
import pickle
import numpy as np
from ucb import ucb 
from thompson import thompson_pf

def experiment(scen, choices, alg="ucb1", num_repeats=1000, num_rounds=1e5, sgd_rounds=100, sgd_stepsize=0.01, mcmc_stepsize=0.5, mcmc_cycles=2,  out_file="out.pickle", log_file="run.log", out_dir="out_test/", verbose=False, gumbel_scale=0.5, gumbel_t_varies=False, greedy_epsilon=0.05, trials_tune=10, record_all=False, add_gumbel=False, exp_type="single-debias"):

    if verbose: print "starting "+ out_file
    num_actions = len(scen) 
    means = np.array([arm.get_expected_reward() for arm in scen])
    best_action = np.argmax(means)

    deltas = means[best_action] - means 
    delta_sum = sum(deltas) 
    inv_delta_sum = 0
    for i in range(len(deltas)):
        if deltas[i] != 0:
            inv_delta_sum += 1.0 / deltas[i]

    #bookkeeping
    bias_tab = np.zeros((num_repeats+trials_tune, num_actions, num_rounds))
    ucb_tab = np.zeros((num_repeats+trials_tune, num_actions, num_rounds))
    mse_tab = np.zeros((num_repeats+trials_tune, num_actions, num_rounds))
    debias_mse_tab = np.zeros((num_repeats+trials_tune, num_actions, sgd_rounds+1))
    debias_bias_tab = np.zeros((num_repeats+trials_tune, num_actions, sgd_rounds+1))
    regret_tab = np.zeros((num_repeats+trials_tune, num_rounds))
    num_pulls_tab = np.zeros((num_repeats+trials_tune, num_actions, num_rounds))
    reward_tab = np.empty((num_repeats+trials_tune, num_actions, num_rounds))
    reward_tab.fill(np.nan)
    reward_per_step_tab = np.empty((num_repeats+trials_tune, num_actions, num_rounds))
    reward_per_step_tab.fill(np.nan)
    accept_ratio_tab = 0

    for trial in range(num_repeats + trials_tune):
        scen_trial = copy.deepcopy(scen)
        if trial % 50 == 0:
            if verbose: print "trial: " + str(trial)

        cumulative_reward = 0
        best_cumulative_reward = 0

        debias = False
        held = False
        rand = None

        if "gumbel" in alg:
            debias = True
            rand = gumbel_r(scale=gumbel_scale)

        elif "debias" in alg:
            debias = True

        elif "held" in alg:
            held = True

        if "ucb1" in alg:
            gen = ucb(scen_trial, rand=rand, gumbel_t_varies=gumbel_t_varies, held=held)

        elif "lil-ucb" in alg:
            gen = ucb(scen_trial, rand=rand, gumbel_t_varies=gumbel_t_varies, held=held, lil_ucb=True)

        elif "thom" in alg:
            gen = thompson_pf(scen_trial, rand=rand, gumbel_t_varies=gumbel_t_varies, held=held)

        elif "greedy" in alg:
            if "e-greedy" in alg:
                gen = ucb(scen_trial, rand=rand, egreedy=True, gumbel_t_varies=gumbel_t_varies, held=held, greedy_epsilon=greedy_epsilon)
            else:
                gen = ucb(scen_trial, rand=rand, greedy=True, gumbel_t_varies=gumbel_t_varies, held=held)


        t = 0 

        for (choice, reward, num_pulls, payoff_sums, ucb_bound, debiaser, num_pulls_held, payoff_sums_held, ind_sample) in gen:

            for arm in range(num_actions):
                num_pulls_tab[trial,:,t] = num_pulls
                if arm == choice:
                    reward_tab[trial, arm, num_pulls[arm]-1] = reward
                    reward_per_step_tab[trial, arm, t] = reward
                    if held:
                        if ind_sample:
                            bias_tab[trial, arm, t] = payoff_sums_held[arm] * 1.0 / num_pulls_held[arm] - means[arm]
                        else:
                            bias_tab[trial, arm, t] = bias_tab[trial, arm, t-1]
                    else:
                        bias_tab[trial, arm, t] = payoff_sums[arm] * 1.0 / num_pulls[arm] - means[arm]
                    if (not held) or (held and ind_sample):
                        bias_tab[trial, arm, t] = bias_tab[trial, arm, t] / scen[arm].get_sigma()
                    mse_tab[trial, arm, t] = bias_tab[trial,arm,t]**2
                else:
                    bias_tab[trial, arm, t] = bias_tab[trial, arm, t-1]
                    mse_tab[trial, arm, t] = mse_tab[trial, arm, t-1]

                if alg == "ucb1" or alg == "lil-ucb":
                    ucb_tab[trial, arm, t] = ucb_bound[arm] / scen[arm].get_sigma()

            cumulative_reward += reward
            best_cumulative_reward += scen_trial[best_action].get_reward()
            regret = best_cumulative_reward - cumulative_reward
            regret_bound = 8 * math.log(t + 5) * inv_delta_sum + (1 + math.pi*math.pi / 3) * delta_sum
            regret_tab[trial, t] = regret
            t += 1

            if t >= num_rounds:
                #regret_all[trial] = regret
                break

        #print "mse", np.linalg.norm(payoff_sums / num_pulls - means)


        # This is the debiasing part, CAUTION: very slow.
        if debias:
            debiaser.setup_sampling()
            #print np.linalg.norm(debiaser.hmu-means)

            accept_ratio_sum = 0.0
            for i in range(sgd_rounds):
                hmu, accept_ratio = debiaser.next(sgd_stepsize=sgd_stepsize, mcmc_stepsize=mcmc_stepsize, cycles=mcmc_cycles)
                accept_ratio_sum += accept_ratio
                for arm in range(num_actions):
                    debias_bias_tab[trial,arm,i] = 1.0 * (debiaser.hmu-means)[arm] / scen[arm].get_sigma()
                    debias_mse_tab[trial,arm,i] = debias_bias_tab[trial,arm,i]**2
            accept_ratio_avg = accept_ratio_sum * 1.0 / sgd_rounds
            if trial > trials_tune:
                accept_ratio_tab += accept_ratio_avg
            debias_bias_tab[trial,:,sgd_rounds] = np.mean(debias_bias_tab[trial,:,-101:-1], axis=1)
            debias_mse_tab[trial,:,sgd_rounds] = debias_bias_tab[trial,:,sgd_rounds]**2
            if verbose: 
                print "trial:" + str(trial)
                print "accept ratio: " + str(accept_ratio_avg)

            #print "sample_mean", debiaser.X[-1,:], payoff_sums / num_pulls
            #print "debiased", debiaser.hmu
            print "improvement", debiaser.hmu - debiaser.X[-1,:]
            if trial < trials_tune:
                mcmc_stepsize = debiaser.tune(mcmc_stepsize, accept_ratio_avg)
                print str(accept_ratio_avg) + " " + str(mcmc_stepsize)

    if verbose: print "regret", regret_tab[:,-1].mean()
    accept_ratio_tab = accept_ratio_tab * 1.0 / num_repeats
     
    pickle_f = open(out_file, "wb")
    

    if exp_type == "single-debias":
        bias_sum = np.sum(bias_tab[trials_tune:,:,:], axis=0)
        ucb_sum = np.sum(ucb_tab[trials_tune:,:,:], axis=0)
        mse_sum = np.sum(mse_tab[trials_tune:,:,:], axis=0)
        debias_bias_sum = np.sum(debias_bias_tab[trials_tune:,:,:], axis=0)
        debias_mse_sum = np.sum(debias_mse_tab[trials_tune:,:,:], axis=0)
        regret_sum = np.sum(regret_tab[trials_tune:,:], axis=0)
        num_pulls_sum = np.sum(num_pulls_tab[trials_tune:,:,-1], axis=0)
        num_pulls_std = np.std(num_pulls_tab[trials_tune:,:,-1], axis=0)
    else:
        bias_mean = np.mean(bias_tab[trials_tune:,:,:], axis=0)
        ucb_mean = np.mean(ucb_tab[trials_tune:,:,:], axis=0)
        mse_mean = np.mean(mse_tab[trials_tune:,:,:], axis=0)
        debias_bias_mean = np.mean(debias_bias_tab[trials_tune:,:,:], axis=0)
        debias_mse_mean = np.mean(debias_mse_tab[trials_tune:,:,:], axis=0)
        regret_mean = np.mean(regret_tab[trials_tune:,:], axis=0)
        num_pulls_mean = np.mean(num_pulls_tab[trials_tune:,:,-1], axis=0)
        num_pulls_std = np.std(num_pulls_tab[trials_tune:,:,-1], axis=0)

    if exp_type == "single-debias":
        plot_params = None
    else:
        plot_params = [np.max(bias_mean), np.min(bias_mean), np.max(mse_mean), np.min(mse_mean), np.max(regret_mean), np.min(regret_mean)]

    if exp_type == "single-debias":
        pickle.dump(plot_params, pickle_f)
        pickle.dump(bias_sum, pickle_f)
        pickle.dump(ucb_sum, pickle_f)
        pickle.dump(mse_sum, pickle_f)
        pickle.dump(debias, pickle_f)
        pickle.dump(debias_bias_sum, pickle_f)
        pickle.dump(debias_mse_sum, pickle_f)
        pickle.dump(regret_sum, pickle_f)
        pickle.dump(mcmc_stepsize, pickle_f)
        pickle.dump(accept_ratio_tab, pickle_f)
        pickle.dump(num_pulls_sum, pickle_f)
        pickle.dump(num_pulls_std, pickle_f)

        log_f = open(log_file, 'a')
        log_f.write(str([(arg, locals()[arg]) for arg in inspect.getargspec(experiment).args])+"\n")
        log_f.write("bias: " + str(bias_sum[:,-1]) + "\n")
        log_f.write("mse: " + str(mse_sum[:,-1]) + "\n")
        log_f.write("regret: " + str(regret_sum[-1]) + "\n")
        log_f.write("debias: " + str(debias)+"\n")
        log_f.write("debiased bias: " + str(debias_bias_sum[:,-1])+ "\n")
        log_f.write("debiased mse: " + str(debias_mse_sum[:,-1]) + "\n")
        log_f.write("mcmc stepsize: " + str(mcmc_stepsize) + "\n")
        log_f.write("accept ratio: " + str(accept_ratio_tab)+"\n")
        log_f.write("num pulls sum: " + str(num_pulls_sum) + "\n")
        log_f.write("num pulls std: " + str(num_pulls_std) + "\n")
        log_f.close()
    else:
        pickle.dump(plot_params, pickle_f)
        pickle.dump(bias_mean, pickle_f)
        pickle.dump(ucb_mean, pickle_f)
        pickle.dump(mse_mean, pickle_f)
        pickle.dump(debias, pickle_f)
        pickle.dump(debias_bias_mean, pickle_f)
        pickle.dump(debias_mse_mean, pickle_f)
        pickle.dump(regret_mean, pickle_f)
        pickle.dump(mcmc_stepsize, pickle_f)
        pickle.dump(accept_ratio_tab, pickle_f)
        pickle.dump(num_pulls_mean, pickle_f)
        pickle.dump(num_pulls_std, pickle_f)

        log_f = open(log_file, 'a')
        log_f.write(str([(arg, locals()[arg]) for arg in inspect.getargspec(experiment).args])+"\n")
        log_f.write("bias: " + str(bias_mean[:,-1]) + "\n")
        log_f.write("mse: " + str(mse_mean[:,-1]) + "\n")
        log_f.write("regret: " + str(regret_mean[-1]) + "\n")
        log_f.write("debias: " + str(debias)+"\n")
        log_f.write("debiased bias: " + str(debias_bias_mean[:,-1])+ "\n")
        log_f.write("debiased mse: " + str(debias_mse_mean[:,-1]) + "\n")
        log_f.write("mcmc stepsize: " + str(mcmc_stepsize) + "\n")
        log_f.write("accept ratio: " + str(accept_ratio_tab)+"\n")
        log_f.write("num pulls mean: " + str(num_pulls_mean) + "\n")
        log_f.write("num pulls std: " + str(num_pulls_std) + "\n")
        log_f.close()

    if record_all:
       pickle.dump(num_pulls_tab[trials_tune:,:,:], pickle_f)
       pickle.dump(bias_tab[trials_tune:,:,:].round(3), pickle_f)
       pickle.dump(reward_tab[trials_tune:,:,:].round(3), pickle_f)
       pickle.dump(reward_per_step_tab[trials_tune:,:,:].round(3), pickle_f)

    pickle_f.close()


    plot_dir = out_dir + "/plots/"
    if not os.path.exists(plot_dir):
        try:
            os.makedirs(plot_dir)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
            else:
                pass

    return plot_dir, plot_params

