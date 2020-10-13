import json
import numpy as np
from collections import defaultdict
from outlier_project.utils.config import *
from pathlib import Path

raw_seed = 567
simulations = 100

performance_dict = {}

method_name = ["SARSA", "sample_average", "categorical", "EM_MLE"]
dist_name = ["G", "T", "E"]

for each_method in method_name:
    performance_dict[each_method] = {}
    for each_dist in dist_name:
        if not ((each_dist == "G") and (each_method == "EM_MLE")):
            performance_dict[each_method][each_dist] = {"S0": {}, "S1": {}}

for each_method in method_name:
    for each_dist in dist_name:
        if not ((each_dist == "G") and (each_method == "EM_MLE")):
            avg_performance_dict = {"S0": [], "S1": []}

            for i in range(simulations):
                with open("../simulation_data/" + "seed_" + str(raw_seed) + "/" + each_method + "/"
                          + each_dist + "/" + str(i) + ".json") as json_file:
                    raw = json.load(json_file)

                per_game_performance_count = {"S0": [], "S1": []}

                for j in range(10, 200):
                    q_table = raw[j]["q_table"]

                    if q_table["S0"]["A0"] > q_table["S0"]["A1"]:
                        per_game_performance_count["S0"].append(1)
                    else:
                        per_game_performance_count["S0"].append(0)

                    if q_table["S1"]["A0"] < q_table["S1"]["A1"]:
                        per_game_performance_count["S1"].append(1)
                    else:
                        per_game_performance_count["S1"].append(0)

                avg_performance_dict["S0"].append(np.average(per_game_performance_count["S0"]))
                avg_performance_dict["S1"].append(np.average(per_game_performance_count["S1"]))

            performance_dict[each_method][each_dist]["S0"]["robust_convergence"] = \
                avg_performance_dict["S0"].count(1) / float(simulations)
            performance_dict[each_method][each_dist]["S1"]["robust_convergence"] = \
                avg_performance_dict["S1"].count(1) / float(simulations)

            performance_dict[each_method][each_dist]["S0"]["average"] = np.average(avg_performance_dict["S0"])
            performance_dict[each_method][each_dist]["S1"]["average"] = np.average(avg_performance_dict["S1"])

            performance_dict[each_method][each_dist]["S0"]["std_error"] = \
                np.std(avg_performance_dict["S0"]) / np.sqrt(simulations)
            performance_dict[each_method][each_dist]["S1"]["std_error"] = \
                np.std(avg_performance_dict["S1"]) / np.sqrt(simulations)

            performance_dict[each_method][each_dist]["S0"]["min"] = np.min(avg_performance_dict["S0"])
            performance_dict[each_method][each_dist]["S1"]["min"] = np.min(avg_performance_dict["S1"])

            performance_dict[each_method][each_dist]["S0"]["max"] = np.max(avg_performance_dict["S0"])
            performance_dict[each_method][each_dist]["S1"]["max"] = np.max(avg_performance_dict["S1"])

data_path = "../results/" + "seed_" + str(raw_seed) + "/"

if not Path(data_path).exists():
    Path(data_path).mkdir(parents=True)
else:
    pass

with open(data_path + "performance.json", 'w', encoding='utf-8') as f:
    json.dump(performance_dict, f, ensure_ascii=False, indent=4)
