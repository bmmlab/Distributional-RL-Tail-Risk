from outlier_project.agents import *
from outlier_project.envs import *
from outlier_project.utils.config import *
from outlier_project.utils.utils import *
import numpy as np


def main():
    configs = Config()

    sims = configs.no_of_simulations
    raw_seed = configs.raw_seed
    np.random.seed(raw_seed)
    seeds = np.random.randint(0, 1000000, size=sims)
    experiment_distribution = ["G", "T", "E"]  # G: gaussian, T: student-t, E: empirical SP500 (1970-2020)

    env = leptokurtosis.LeptokurticEnv()

    env.true_mean = configs.true_mean
    env.df = configs.df

    save_configuration_to_json(c=configs, seed=raw_seed)

    for each_sim in range(sims):
        for each_dist in experiment_distribution:
            env.reward_distribution_type = each_dist

            env.seed(seed=seeds[each_sim])  # fix seed
            env.initialize_reward_distribution()

            # ---------------------------------SARSA-------------------------------------------------
            res_dict = traditional_SARSA.sarsa(env=env, c=configs)
            save_simulation_data_to_json(which_method="SARSA",
                                         which_dist=each_dist,
                                         which_simulation=each_sim,
                                         result_dict=res_dict,
                                         seed=raw_seed)

            # ----------------------------categorical (-30, 30) with 100 bins-------------------------
            res_dict = categorical_TD.categorical(env=env, c=configs)
            save_simulation_data_to_json(which_method="categorical",
                                         which_dist=each_dist,
                                         which_simulation=each_sim,
                                         result_dict=res_dict,
                                         seed=raw_seed)

            # -----------------------distributional RL------------------------------------------------
            configs.efficient_estimation_method = "sample_average"
            res_dict = bmm_distributional_RL.distributional_sarsa(env=env, c=configs)
            save_simulation_data_to_json(which_method=configs.efficient_estimation_method,
                                         which_dist=each_dist,
                                         which_simulation=each_sim,
                                         result_dict=res_dict,
                                         seed=raw_seed)

            if each_dist != "G":  # MLE in gaussian is the same as sample average
                configs.efficient_estimation_method = "EM_MLE"
                res_dict = bmm_distributional_RL.distributional_sarsa(env=env, c=configs)

                save_simulation_data_to_json(which_method=configs.efficient_estimation_method,
                                             which_dist=each_dist,
                                             which_simulation=each_sim,
                                             result_dict=res_dict,
                                             seed=raw_seed)

        print("sim " + str(each_sim) + " done")


if __name__ == '__main__':
    main()
