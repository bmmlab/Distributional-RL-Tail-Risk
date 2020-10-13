# -*- coding:utf-8 -*-
#######################################################################
# Copyright (C) 2018 -                                                #
# Shijie Huang (harveyh@student.unimelb.edu.au)                       #
# Brain, Mind and Markets Laboratory                                  #
# https://research.unimelb.edu.au/brain-mind-markets                  #
# University of Melbourne                                             #
#######################################################################

import numpy as np
from collections import deque
import pandas as pd
from outlier_project.utils.config import *
from outlier_project.utils.policies import *
from outlier_project.utils.utils import *
from outlier_project.envs import leptokurtosis
from copy import deepcopy


def sarsa(env, c):
    alpha_lr = c.alpha_lr  # learning rate
    gamma_dr = c.gamma_dr  # discount rate
    stop_explore = c.stop_explore  # stop explore episodes

    states = c.states
    actions = c.actions

    states_keys = list(states.keys())
    actions_keys = list(actions.keys())

    q_table = pd.DataFrame(data=0.0, index=actions_keys, columns=states_keys, dtype=np.float)  # initialize q table

    action_value_dist = initialize_distribution(states=states_keys, actions=actions_keys)

    history = deque([])  # record data for statistical analysis

    for each_episode in range(0, c.total_episode):
        next_state = env.state  # initialize state
        reward_list = deque()  # initialize steps rewards history for traditional SARSA

        for each_step in range(0, c.steps):
            current_state = next_state  # update the current states

            action_key = e_greedy(state=current_state, action_list=actions_keys,
                                  q_table=q_table, episode=each_episode,
                                  stop_explore=stop_explore)

            next_state, reward, done, _ = env.step(action=actions[action_key])

            # core SARSA algorithm
            q_value = q_table[current_state][action_key]
            q_value_next = q_table[next_state].max()
            td_delta = reward + gamma_dr * q_value_next - q_value

            q_value += alpha_lr * td_delta  # calculate new action value

            q_table.at[action_key, current_state] = q_value  # update action value in the q table

            action_value_dist[current_state][action_key].append(q_value)  # add reward to reward list

            reward_list.append(reward)

        # record important history per episode
        history.append(
            {
                "episode": each_episode,
                "current_state": current_state,
                "current_action": action_key,
                "td_delta": td_delta,
                "q_table": q_table.copy().to_dict(),
                # "reward_distribution": deepcopy(reward_dist)
            }
        )

        # print(q_table)

    return list(history)
