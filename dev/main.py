from optparse import OptionParser
import os.path
from time import gmtime, strftime
import errno    
import glob
from plot import plot
import copy as cp
import shutil
import pdb
from experiment import experiment
import os
from reward_distr import Gaussian, Bernoulli
from scenarios import get_scenario

def run(num_repeats=1000, num_rounds=1000, sgd_rounds=100, sgd_stepsize=0.01, mcmc_cycles=30, mcmc_stepsize=0.5, mu_scale=0.5, choice="all",distr="gauss", verbose=False, alg="greedy", exp_type="single-debias", gumbel_scale=1.0, gumbel_t_varies=False, greedy_epsilon=0.05, output_dir="out", record_all=False, trials_tune=20, add_gumbel=False, div=1):
    scenario = get_scenario(distr, choice, mu_scale)
    out_dir = output_dir + "/" + str(exp_type) 
    if alg  == "e-greedy":
        alg_str = alg + "_" + str(greedy_epsilon)
    else:
        alg_str = alg 

    if exp_type == "single-debias":
        out_dir = out_dir + "/" + distr + "/" + choice + "_" + str(mu_scale) + "/" + alg_str + "/" + str(num_rounds) 
        file_str = distr + "_" + choice + "_" + str(mu_scale) + "_" + str(num_rounds) + "_" + str(gumbel_scale) + "_" + str(div)

    if exp_type == "vary-mu":
        out_dir = out_dir + "/" + alg_str + "/" + distr + "/" + choice + "/" + str(mu_scale)
        file_str = distr + "_" + choice + "_" + str(mu_scale)

    plot_params_all = [-1 * float("inf"), float("inf"), -1 * float("inf"), float("inf"), -1 * float("inf"), float("inf")]
    out_file_all = {}
    plot_dir_all = {}
    file_str_all = {}

    if exp_type == "vary-mu":
        alg_list = [alg_str]
    else:
        if "thom" in alg_str or "e-greedy" in alg_str:
            if add_gumbel:
                alg_list = [alg_str+"-gumbel"]
            else:
                alg_list = [alg_str, alg_str+"-debias", alg_str+"-held"]
        else:
            alg_list = [alg_str, alg_str+"-gumbel", alg_str+"-held"]

    for a in alg_list:

        file_str_a = file_str + "_" + a 
        out_file = out_dir + "/" + file_str_a + ".pickle"
        log_file = out_dir + "/" + file_str_a + ".log"

        if os.path.exists(out_file):
            try:
                os.remove(out_file)
            except:
                print out_file
                print div
                raise
        if os.path.exists(log_file):
            os.remove(log_file)

        if not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
                pass

        plot_dir, plot_params = experiment(scenario, choice, alg=a, num_repeats=num_repeats, num_rounds=num_rounds, out_file=out_file, log_file=log_file, out_dir=out_dir, verbose=verbose, sgd_rounds=sgd_rounds, mcmc_stepsize=mcmc_stepsize, mcmc_cycles=mcmc_cycles, sgd_stepsize=sgd_stepsize, gumbel_scale=gumbel_scale, gumbel_t_varies=gumbel_t_varies, greedy_epsilon=greedy_epsilon, record_all=record_all, trials_tune=trials_tune, add_gumbel=add_gumbel, exp_type=exp_type) 

        if exp_type == "vary-mu":
            for i in range(len(plot_params)):
                if i % 2== 0:
                    plot_params_all[i] = max(plot_params_all[i], plot_params[i])
                else:
                    plot_params_all[i] = min(plot_params_all[i], plot_params[i])

            out_file_all[a] = out_file
            plot_dir_all[a] = plot_dir
            file_str_all[a] = file_str_a

    if exp_type == "vary-mu":
        for a in alg_list:
            plot(scenario, choice, a, out_file_all[a], plot_dir_all[a], plot_params_all, file_str_all[a])

if __name__ == "__main__":
    init_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    parser = OptionParser()
    parser.add_option("-s", "--choice", action="store", dest="choice", type="string", default="all", help="Choose from all, 2-best, 2-bw, 3-best, 3-mid")
    parser.add_option("-a", "--alg", action="store", dest="alg", type="string", default="greedy", help="Choose from greedy, thom, lil-ucb, e-greedy")
    parser.add_option("-v", action="store_true", dest="verbose", default=False, help="Set verbose to True")
    parser.add_option("-q", action="store_false", dest="verbose", help="Set verbose to False")
    parser.add_option("-r", "--numrepeats", action="store", dest="num_repeats", type="int", default=1000, help="Number of repeats")
    parser.add_option("-t", "--numrounds", action="store", dest="num_rounds", type="int", default=40, help="Number of rounds/horizon for each of the repeating trials")
    parser.add_option("-g", "--sgdrounds", action="store", dest="sgd_rounds", type="int", default=100, help="Number of SGD (stochastic gradient descent) steps")
    parser.add_option("-z", "--sgdstepsize", action="store", dest="sgd_stepsize", type="float",default=0.01, help="Step size of SGD")
    parser.add_option("-m", "--mcmccycles", action="store", dest="mcmc_cycles",type="int", default=30, help="Number of MCMC steps")
    parser.add_option("-c", "--mcmcstepsize", action="store", dest="mcmc_stepsize",type="float", default=0.5, help="MCMC step size")
    parser.add_option("-e", "--experiment", action="store", dest="exp_type", type="string", default="single-debias", help="Type of experiment. Choose from single-debias for debiasing experiments, or vary-mu for general experiments without debiasing")
    parser.add_option("-x", "--muscale", action="store", dest="mu_scale", type="float", default=0.5, help="The scale of how far apart the means of the arms are from each other. Default is 0.5. Details see scenarios.py")
    parser.add_option("-b", "--gumbelscale", action="store", dest="gumbel_scale", type="float", default=1.0, help="The scale parameter for the added Gumbel noise.")
    parser.add_option("-w", "--greedyepsilon", action="store", dest="greedy_epsilon", type="float", default=0.05, help="Epsilon in epsilon-Greedy")
    parser.add_option("-f",  action="store_true", dest="gumbel_t_varies", default=False, help="Set True for varying the Gumbel noise magnitude with time")
    parser.add_option("-p", action="store_false", dest="gumbel_t_varies", help="Set False for varying the Gumbel noise magnitude with time")
    parser.add_option("-l", action="store_true", default=False, dest="record_all", help="Set True for logging data at every round")
    parser.add_option("-d", "--distr", action="store", dest="distr", type="string", default="gauss", help="The distribution for each arm. Only available choice currently: gauss")
    parser.add_option("-u", "--tune", action="store", dest="trials_tune", type="int",default=20, help="The number of MCMC steps used to automatically tune the MCMC step size based on the acceptance ratio")
    parser.add_option("-y", action="store_true", default=False, dest="add_gumbel", help="Set True for adding Gumbel noise to all rewards")
    parser.add_option("-j", action="store_false", dest="add_gumbel", help="Set False for adding Gumbel noise to all rewards")
    parser.add_option("-k", action="store", dest="div", type="int", default=1, help="Divison index. For faster experimental time, we divide up the trials, each specified by a div index and the number of repeats in that particular division. See process_div.py to combine results from such divisions.")
    parser.add_option("-o", "--out", action="store", dest="output_dir", type="string", default="out", help="The file location for the output directory")
    (options, args) = parser.parse_args() 
    run(num_repeats=options.num_repeats, num_rounds=options.num_rounds,sgd_rounds=options.sgd_rounds, sgd_stepsize=options.sgd_stepsize, mcmc_cycles=options.mcmc_cycles, mcmc_stepsize=options.mcmc_stepsize, choice=options.choice, verbose=options.verbose, exp_type=options.exp_type, mu_scale=options.mu_scale, alg=options.alg, gumbel_scale=options.gumbel_scale, gumbel_t_varies=options.gumbel_t_varies, greedy_epsilon=options.greedy_epsilon, output_dir=options.output_dir, distr=options.distr, record_all=options.record_all, trials_tune=options.trials_tune, add_gumbel=options.add_gumbel, div=options.div)
    print init_time
    print strftime("%Y-%m-%d %H:%M:%S", gmtime())
