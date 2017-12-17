import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random
import os
import errno
import numpy as np
import pickle
import pdb
import re

def plot(scen, choices, alg, in_file, out_dir, plot_params, file_str, debias_only=False):

    f = open(in_file, 'r')

    plot_params_file = pickle.load(f)
    bias_mean = pickle.load(f)
    ucb_mean = pickle.load(f)
    mse_mean = pickle.load(f)
    debias = pickle.load(f)
    debias_bias_mean = pickle.load(f)
    debias_mse_mean = pickle.load(f)
    regret_mean = pickle.load(f)

    f.close()
    
    if not debias_only:
        out_dir_bias = out_dir + "/bias"
        if not os.path.exists(out_dir_bias):
            try:
                os.makedirs(out_dir_bias)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
                else:
                    pass

        out_dir_mse= out_dir + "/mse"
        if not os.path.exists(out_dir_mse):
            try:
                os.makedirs(out_dir_mse)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
                else:
                    pass

        out_dir_regret = out_dir + "/regret"
        if not os.path.exists(out_dir_regret):
            try:
                os.makedirs(out_dir_regret)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
                else:
                    pass

    if debias:
        out_dir_debias = out_dir+ "/debiased"
        if not os.path.exists(out_dir_debias):
            try:
                os.makedirs(out_dir_debias)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
                else:
                    pass

    num_actions, num_rounds = bias_mean.shape 
    sgd_rounds = debias_bias_mean.shape[1]

    arms_to_plot = range(num_actions)

    if not debias_only:
        color=iter(plt.cm.Set3(np.linspace(0,1,len(arms_to_plot))))

        for arm in arms_to_plot:
            c=next(color)
            plt.plot(range(num_rounds), bias_mean[arm], color=c, linestyle='-', label="mu: " + str(scen[arm].get_expected_reward(precision=2)))
        plt.legend()
        plt.xlabel("t: round")
        plt.ylabel("relative bias")
        plt.ylim([plot_params[1], plot_params[0]])
        plt.savefig(out_dir_bias+ "/" + file_str + "_bias.png")
        plt.clf()

        if alg == "ucb1" or alg == "lil-ucb":
            color=iter(plt.cm.Set3(np.linspace(0,1,len(arms_to_plot))))
            for arm in arms_to_plot:
                c=next(color)
                plt.plot(range(num_rounds), bias_mean[arm], color=c, linestyle='-', label="mu: " + str(scen[arm].get_expected_reward(precision=2)))
                plt.plot(range(num_rounds), -1*ucb_mean[arm], color=c, linestyle='--')
                plt.plot(range(num_rounds), ucb_mean[arm], color=c, linestyle='--')
            plt.legend()
            plt.xlabel("t: round")
            plt.ylabel("relative bias")
            plt.savefig(out_dir_bias+ "/" + file_str + "_bias_ucb.png")
            plt.clf()

        color=iter(plt.cm.Set3(np.linspace(0,1,len(arms_to_plot))))
        for arm in arms_to_plot:
            c=next(color)
            plt.plot(range(num_rounds), mse_mean[arm], color=c, linestyle='-', label="mu:" + str(scen[arm].get_expected_reward(precision=2)))

        plt.legend()
        plt.xlabel("t: round")
        plt.ylabel("mean squared error")
        plt.ylim([plot_params[3], plot_params[2]])
        plt.savefig(out_dir_mse+ "/" + file_str + "_mse.png")
        plt.clf()

        plt.plot(range(num_rounds), regret_mean, linestyle='-')

        plt.xlabel("t: round")
        plt.ylabel("regret")
        plt.ylim([plot_params[5], plot_params[4]])
        plt.savefig(out_dir_regret+ "/" + file_str + "_regret.png")
        plt.clf()

    if debias:
        color=iter(plt.cm.Set3(np.linspace(0,1,len(arms_to_plot))))
        for arm in arms_to_plot:
            c=next(color)
            plt.plot(range(sgd_rounds), debias_bias_mean[arm], color=c, linestyle='-', label="mu:" + str(scen[arm].get_expected_reward(precision=2)))

        plt.legend()
        plt.xlabel("t: sgd round")
        plt.ylabel("relative bias")
        plt.ylim([plot_params[1], plot_params[0]])
        plt.savefig(out_dir_debias+ "/" + file_str + "_bias.png")
        plt.clf()

        if not debias_only:
            color=iter(plt.cm.Set3(np.linspace(0,1,len(arms_to_plot))))
            for arm in arms_to_plot:
                c=next(color)
                plt.plot(range(sgd_rounds), debias_mse_mean[arm], color=c, linestyle='-', label="mu:" + str(scen[arm].get_expected_reward(precision=2)))

            plt.legend()
            plt.xlabel("t: sgd round")
            plt.ylabel("mean squared error")
            plt.ylim([plot_params[3], plot_params[2]])
            plt.savefig(out_dir_debias+ "/" + file_str + "_mse.png")
            plt.clf()
