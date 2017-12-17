from reward_distr import Gaussian, Bernoulli

num_arms=5
# choices: all, 2-best, 2-bw, 3-best, 3-mid
def get_scenario(distr="gauss", choices="all", mu_scale=1):
    scenario_all = []
    scenario = []
    mus = [2.0,1.5,1.0, 0.75, 0.5]
    if distr == "gauss":
        for mu in mus:
            scenario_all.append(Gaussian(mu * mu_scale, 1.0, 0, 5.0))
    if choices == "2-best":
        scenario = scenario_all[:2]
    elif choices == "2-bw":
        scenario.append(scenario_all[0])
        scenario.append(scenario_all[-1])
    elif choices == "3-best":
        scenario = scenario_all[:3]
    elif choices == "3-mid":
        scenario.append(scenario_all[0])
        scenario.append(scenario_all[num_arms/2])
        scenario.append(scenario_all[-1])
    elif choices == "all":
        scenario = scenario_all
    return scenario
