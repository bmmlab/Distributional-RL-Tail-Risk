import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from copy import deepcopy

raw_seed = 567
simulations = 100

# -----------------------------------figure 4------------------------------------------
agents = ["SARSA", "EM_MLE"]
figure_4 = {}

extract_simulation_id = 7  # from 0 to 99
for each_agent in agents:
    figure_4[each_agent] = {"S0": {"A0": [], "A1": []}, "S1": {"A0": [], "A1": []}}

    with open("../simulation_data/" + "seed_" + str(raw_seed) + "/" + each_agent + "/T/" + str(
            extract_simulation_id) + ".json") as json_file:
        raw = json.load(json_file)

    for j in range(10, 200):
        current_state = raw[j]["current_state"]
        current_action = raw[j]["current_action"]

        figure_4[each_agent][current_state][current_action].append(raw[j]["td_delta"])

with open("../results/" + "seed_" + str(raw_seed) + "/" + "figure_4.json", 'w', encoding='utf-8') as f:
    json.dump(figure_4, f, ensure_ascii=False, indent=4)

# -----------------------------------figure 5------------------------------------------
agents = ["SARSA", "sample_average", "EM_MLE"]
figure_5 = {}
for each_agent in agents:
    figure_5[each_agent] = {"S0": {"A0": [], "A1": []}, "S1": {"A0": [], "A1": []}}

    for each_simulation in range(simulations):
        with open("../simulation_data/" + "seed_" + str(raw_seed) + "/" + each_agent + "/T/" + str(
                each_simulation) + ".json") as json_file:
            raw = json.load(json_file)

        q_table = raw[-1]["q_table"]

        for p in ("S0", "S1"):
            for q in ("A0", "A1"):
                figure_5[each_agent][p][q].append(q_table[p][q])

with open("../results/" + "seed_" + str(raw_seed) + "/" + "figure_5.json", 'w', encoding='utf-8') as f:
    json.dump(figure_5, f, ensure_ascii=False, indent=4)

    # -----------------------------------figure 6------------------------------------------
figure_6 = {}
extract_simulation_id = 5  # \in [0.. 99]

# # observe
# for i in range(10, 200):
#     print(raw[i]["episode"])
#     print(raw[i]["q_table"])


extract_episode_ID = [16, 17]  # \in [10.. 199]

with open("../simulation_data/" + "seed_" + str(raw_seed) + "/categorical/T/" + str(
        extract_simulation_id) + ".json") as json_file:
    raw = json.load(json_file)

for episode in extract_episode_ID:
    figure_6["episode_" + str(episode)] = deepcopy(raw[episode]["q_distribution"])

with open("../results/" + "seed_" + str(raw_seed) + "/" + "figure_6.json", 'w', encoding='utf-8') as f:
    json.dump(figure_6, f, ensure_ascii=False, indent=4)

# -----------------------------------Plots------------------------------------------
# -----------------------------------figure 4---------------------------------------
with open("../results/" + "seed_" + str(raw_seed) + "/" + "figure_4.json") as f:
    raw = json.load(f)

for i in ("S0", "S1"):
    for j in ("A0", "A1"):
        sns.set_style("ticks")
        sub_color = (0.95, 0.95, 0.95)
        plt.rcParams['figure.figsize'] = [10, 3.5]

        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        font_style = {"fontweight": "bold", "fontsize": 14}
        # We can set the number of bins with the `bins` kwarg
        sns.distplot(raw["SARSA"][i][j], bins=50, kde=False, rug=True, ax=axs[0], color='#4ca84c',
                     hist_kws={"alpha": 1}, rug_kws={"alpha": 0.2, 'color': 'black', 'height': '0.02'})
        sns.distplot(raw["EM_MLE"][i][j], bins=50, kde=False, rug=True, ax=axs[1], color='#4c4ca8',
                     hist_kws={"alpha": 1}, rug_kws={"alpha": 0.2, 'color': 'black', 'height': '0.02'})
        axs[0].set_xlabel('TD Errors', font_style)
        axs[0].tick_params(labelsize=14)
        axs[1].tick_params(labelsize=14)
        axs[1].set_xlabel('TD Errors', font_style)
        axs[0].set_ylabel('Frequency', font_style)
        axs[0].set_title('TD', font_style)
        axs[1].set_title('e-disRL', font_style)

        fig.set_facecolor(sub_color)
        axs[0].set_facecolor(sub_color)
        axs[1].set_facecolor(sub_color)
        sns.despine()
        fig.savefig("../results/seed_" + str(raw_seed) + "/" + f"Fig3{i}{j}.pdf")

# -----------------------------------figure 5---------------------------------------
with open("../results/" + "seed_" + str(raw_seed) + "/" + "figure_5.json") as f:
    raw = json.load(f)

for i in ("S0", "S1"):
    for j in ("A0", "A1"):
        sns.set_style("ticks")
        sub_color = (0.95, 0.95, 0.95)
        plt.rcParams['figure.figsize'] = [10, 3.5]

        font_style = {"fontweight": "bold", "fontsize": 14}
        fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)

        # We can set the number of bins with the `bins` kwarg
        sns.distplot(raw["SARSA"][i][j], bins=50, kde=False, rug=True, ax=axs[0], color='#4ca84c',
                     hist_kws={"alpha": 1}, rug_kws={"alpha": 0.2, 'color': 'black', 'height': '0.02'})
        sns.distplot(raw["sample_average"][i][j], bins=50, kde=False, rug=True, ax=axs[1], color='#c75c32',
                     hist_kws={"alpha": 1}, rug_kws={"alpha": 0.2, 'color': 'black', 'height': '0.02'})
        sns.distplot(raw["EM_MLE"][i][j], bins=50, kde=False, rug=True, ax=axs[2], color='#4c4ca8',
                     hist_kws={"alpha": 1}, rug_kws={"alpha": 0.2, 'color': 'black', 'height': '0.02'})

        axs[0].tick_params(labelsize=14)
        axs[1].tick_params(labelsize=14)
        axs[2].tick_params(labelsize=14)
        axs[0].set_xlabel('Q-values', font_style)
        axs[1].set_xlabel('Q-values', font_style)
        axs[2].set_xlabel('Q-values', font_style)
        axs[0].set_ylabel('Frequency', font_style)
        axs[0].set_title('TD', font_style)
        axs[1].set_title('e-disRL-', font_style)
        axs[2].set_title('e-disRL', font_style)

        fig.set_facecolor(sub_color)
        axs[0].set_facecolor(sub_color)
        axs[1].set_facecolor(sub_color)
        axs[2].set_facecolor(sub_color)

        sns.despine()
        fig.savefig("../results/seed_" + str(raw_seed) + "/" + f"Fig4{i}{j}.pdf")

# -----------------------------------figure 6---------------------------------------
with open("../simulation_data/" + "seed_" + str(raw_seed) + "/configuration.json") as f:
    categorical_config = json.load(f)

x_axis_data = np.linspace(categorical_config["v_min"], categorical_config["v_max"], categorical_config["no_of_atoms"])

with open("../results/" + "seed_" + str(raw_seed) + "/" + "figure_6.json") as f:
    raw = json.load(f)

for i in ("S0", "S1"):
    for j in ("A0", "A1"):
        sns.set_style("ticks")
        sub_color = (0.95, 0.95, 0.95)
        plt.rcParams['figure.figsize'] = [10, 2]

        font_style = {"fontweight": "bold", "fontsize": 14}
        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

        # We can set the number of bins with the `bins` kwarg
        axs[0].bar(x_axis_data, raw["episode_16"][i][j], 0.15, color='#c70851')

        axs[1].bar(x_axis_data, raw["episode_17"][i][j], 0.15, color='#c70851')

        axs[0].tick_params(labelsize=14)
        axs[1].tick_params(labelsize=14)
        axs[0].set_ylim(0, 1)
        axs[1].set_ylim(0, 1)

        # manually change the plotting xlims
        if (i == "S0") and (j == "A0"):
            axs[0].set_xlim(10, 17)
            axs[1].set_xlim(-3.5, 3.5)
        elif (i == "S0") and (j == "A1"):
            axs[0].set_xlim(-30.1, -25)
            axs[1].set_xlim(-30.1, -25)
        elif (i == "S1") and (j == "A0"):
            axs[0].set_xlim(-30.1, -25)
            axs[1].set_xlim(-3.5, 3.5)
        else:
            axs[0].set_xlim(10, 17)
            axs[1].set_xlim(-30.1, -25)

        axs[0].set_xlabel('Q-value histogram', font_style)
        axs[1].set_xlabel('Q-value histogram', font_style)
        axs[0].set_ylabel('Probability', font_style)
        axs[0].set_title('Episode 16', font_style)
        axs[1].set_title('Episode 17', font_style)

        fig.set_facecolor(sub_color)
        axs[0].set_facecolor(sub_color)
        axs[1].set_facecolor(sub_color)
        sns.despine()
        fig.savefig("../results/seed_" + str(raw_seed) + "/" + f"Fig5{i}{j}.pdf")
