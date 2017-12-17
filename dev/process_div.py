import os
import numpy as np
import pdb
import pickle
from optparse import OptionParser
def process_div(input_dir, exp_type="single-debias", distr="gauss", alg="lil-ucb", eps=0.1, choice="all", num_rounds=40, gumbel_scale=1.0, mu_scale=0.5, divs=10, num_repeats=1000):
    in_dir = input_dir + "/" + str(exp_type) 
    if alg == "e-greedy":
        alg = alg + "_" + str(eps)

    in_dir_cp = in_dir + "/" + distr + "/" + choice + "_" + str(mu_scale) + "/" + alg + "/" + str(num_rounds) 
    file_str = distr + "_" + choice + "_" + str(mu_scale) + "_" + str(num_rounds) + "_" + str(gumbel_scale)
    file_str_tally = distr + "_" + choice + "_" + str(mu_scale) + "_" + str(num_rounds) + "_" + str(gumbel_scale) + "_" + alg 

    tally_pickle_files(in_dir_cp, file_str, file_str_tally, alg, divs, num_repeats)

def tally_pickle_files(in_dir, file_str, file_str_tally, alg, divs, num_repeats):
    for addi in ["", "-held", "-gumbel", "-debias"]:
        pickle_list = []
        for div in range(1,divs+1):
            file_str_addi = file_str + "_" + str(div) + "_" + alg + addi + ".pickle"
            pickle_file = in_dir + "/" + file_str_addi
            #print pickle_file
            if os.path.exists(pickle_file):
                pickle_list.append(pickle_file)
        bias_sum_tally = None
        ucb_sum_tally  = None
        mse_sum_tally = None
        debias_bias_sum_tally = None
        debias_mse_sum_tally = None
        regret_sum_tally = None
        num_pulls_sum_tally = None
        for pickle_file in pickle_list:
            f = open(pickle_file, "r")
            plot_params = pickle.load(f)
            bias_sum = pickle.load(f)
            ucb_sum = pickle.load(f)
            mse_sum = pickle.load(f)
            debias = pickle.load(f)
            debias_bias_sum = pickle.load(f)
            debias_mse_sum = pickle.load(f)
            regret_sum = pickle.load(f)
            mcmc_stepsize = pickle.load(f)
            accept_ratio_tab = pickle.load(f)
            num_pulls_sum = pickle.load(f)
            num_pulls_std = pickle.load(f)
            f.close()

            bias_sum_tally = update_tally(bias_sum_tally, bias_sum)
            ucb_sum_tally = update_tally(ucb_sum_tally, ucb_sum)
            mse_sum_tally = update_tally(mse_sum_tally, mse_sum)
            debias_bias_sum_tally = update_tally(debias_bias_sum_tally, debias_bias_sum)
            debias_mse_sum_tally = update_tally(debias_mse_sum_tally, debias_mse_sum)
            regret_sum_tally = update_tally(regret_sum_tally, regret_sum)
            num_pulls_sum_tally = update_tally(num_pulls_sum_tally, num_pulls_sum)

        if len(pickle_list) > 0:
            bias_mean = bias_sum_tally * 1.0 / num_repeats 
            ucb_mean = ucb_sum_tally * 1.0 / num_repeats 
            mse_mean = mse_sum_tally * 1.0 / num_repeats 
            debias_bias_mean = debias_bias_sum_tally * 1.0 / num_repeats 
            debias_mse_mean = debias_mse_sum_tally * 1.0 / num_repeats 
            regret_mean = regret_sum_tally * 1.0/num_repeats 
            num_pulls_mean = num_pulls_sum_tally * 1.0 / num_repeats 

            if debias:
                plot_params = [max(np.max(bias_mean), np.max(debias_bias_mean)), min(np.min(bias_mean), np.min(debias_bias_mean)), max(np.max(mse_mean), np.max(debias_mse_mean)), min(np.min(mse_mean), np.min(debias_mse_mean)), np.max(regret_mean), np.min(regret_mean)]
            pickle_tally = in_dir + "/" + file_str_tally + addi + ".pickle"
            f = open(pickle_tally, "wb")
            pickle.dump(plot_params, f)
            pickle.dump(bias_mean, f)
            pickle.dump(ucb_mean, f)
            pickle.dump(mse_mean, f)
            pickle.dump(debias, f)
            pickle.dump(debias_bias_mean, f)
            pickle.dump(debias_mse_mean, f)
            pickle.dump(regret_mean, f)
            pickle.dump(mcmc_stepsize, f)
            pickle.dump(accept_ratio_tab, f)
            pickle.dump(num_pulls_mean, f)
            pickle.dump(num_pulls_std, f)
            f.close()

            log_file = in_dir + "/" + file_str_tally + addi + ".log"
            log_f = open(log_file, 'w')
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

def update_tally(tally, div):
    if tally is None:
        return div
    else:
        return tally + div

def main():
    parser = OptionParser()
    parser.add_option("-i", "--input", action="store", dest="input_dir", type="string", default="../out", help="Input directory to consolidate experiment outputs")
    parser.add_option("-a", "--alg", action="store", dest="alg", type="string", default="lil-ucb", help="The bandit algorithm used for the experiments")
    parser.add_option("-x", "--muscale", action="store", dest="mu_scale", type="float", default=0.5, help="The scale of how far apart the means of the arms are from each other. Default is 0.5. Details see scenarios.py")
    parser.add_option("-b", "--gumbelscale", action="store", dest="gumbel_scale", type="float", default=1.0, help="The scale parameter for the added Gumbel noise")
    parser.add_option("-s", "--choice", action="store", dest="choice", type="string", default="all", help="Choose from all, 2-best, 2-bw, 3-best, 3-mid")
    parser.add_option("-t", "--numrounds", action="store", dest="num_rounds", type="int", default=40, help="Number of rounds/horizon for each of the repeating trials")
    parser.add_option("-w", "--greedyepsilon", action="store", dest="greedy_epsilon", type="float", default=0.05, help="Epsilon in epsilon-Greedy")
    parser.add_option("-e", "--experiment", action="store", dest="exp_type", type="string", default="single-debias", help="Type of experiment. Choose from single-debias for debiasing experiments, or vary-mu for general experiments without debiasing")
    parser.add_option("-d", "--distr", action="store", dest="distr", type="string", default="gauss", help="The distribution for each arm. Only available choice currently: gauss")
    parser.add_option("-k", action="store", dest="divs", type="int", default=1, help="Number of divisions to collect experiment output from")
    parser.add_option("-r", "--numrepeats", action="store", dest="num_repeats", type="int", default=1000, help="Number of repeats")

    (options, args) = parser.parse_args() 
    process_div(options.input_dir, exp_type=options.exp_type, distr=options.distr, alg=options.alg, eps=options.greedy_epsilon, choice=options.choice, num_rounds=options.num_rounds, gumbel_scale=options.gumbel_scale, mu_scale=options.mu_scale, divs=options.divs, num_repeats=options.num_repeats)

if __name__ == "__main__":
    main()
