# -*- coding:utf-8 -*-
#######################################################################
# Copyright (C) 2018 -                                                #
# Implemented by Shijie Huang (harveyh@student.unimelb.edu.au)        #
# Brain, Mind and Markets Laboratory (http://bmmlab.org/)             #
# University of Melbourne                                             #
#                                                                     #
# Bellemare, M. G., Dabney, W., & Munos, R. (2017).                   #
# A distributional perspective on reinforcement learning.             #
# In Proceedings of the 34th ICML-Volume 70 (pp. 449-458)             #
#######################################################################

import numpy as np
from collections import deque
import pandas as pd
from outlier_project.utils.config import *
from outlier_project.utils.policies import *
from outlier_project.utils.utils import *
from outlier_project.envs import leptokurtosis
from copy import deepcopy


def categorical(env, c):
    gamma_dr = c.gamma_dr  # discount rate
    stop_explore = c.stop_explore  # stop explore episodes

    states = c.states
    actions = c.actions

    states_keys = list(states.keys())
    actions_keys = list(actions.keys())

    atoms = np.linspace(c.v_min, c.v_max, c.no_of_atoms)  # Z
    # {z_i = Vmin + i\Delta_z: 0 <= i < N}
    delta_z = (c.v_max - c.v_min) / float(c.no_of_atoms - 1)  # bin size

    q_table = pd.DataFrame(data=0.0, index=actions_keys, columns=states_keys, dtype=np.float)  # initialize q table

    q_dist = initialize_record_distribution(states=states, actions=actions, empty_atoms=atoms)

    history = deque([])  # record data for statistical analysis
    reward_list = deque()

    for each_episode in range(0, c.total_episode):
        next_state = env.state  # initialize state

        for each_step in range(0, c.steps):
            # update the current states
            current_state = next_state

            action_key = e_greedy(state=current_state, action_list=actions_keys,
                                  q_table=q_table, episode=each_episode,
                                  stop_explore=stop_explore)

            next_state, reward, done, _ = env.step(action=actions[action_key])
            action_key_next = q_table[next_state].idxmax()
            histogram = q_dist[next_state][action_key_next].copy()

            # distributional update
            atoms_next = reward + gamma_dr * (1 - done) * atoms
            atoms_next = np.clip(atoms_next, c.v_min, c.v_max)  # clamp the range of atoms

            # recalculate the probability histogram
            b = (atoms_next - c.v_min) / delta_z
            l = np.floor(b).astype(int)  # floor
            u = np.ceil(b).astype(int)  # ceilings

            d_m_l = (u + (l == u) - b) * histogram
            d_m_u = (b - l) * histogram

            target_histogram = np.zeros(c.no_of_atoms)  # initialize the target distribution m

            # re-distribute the distribution to the target distribution according to index l and u
            np.add.at(target_histogram, l, d_m_l)
            np.add.at(target_histogram, u, d_m_u)
            q_dist[current_state][action_key] = target_histogram.copy().tolist()

            # calculate the new value and put it into the corresponding distribution
            # q_table[current_state][action_key] = np.sum(target_histogram * (atoms + delta_z / 2.0))
            q_table[current_state][action_key] = np.sum(target_histogram * atoms)

            reward_list.append(reward)

        # record important history per episode
        history.append(
            {
                "episode": each_episode,
                "current_state": current_state,
                "current_action": action_key,
                "q_table": q_table.copy().to_dict(),
                "q_distribution": deepcopy(q_dist)
                # "reward_distribution": deepcopy(reward_dist)
            }
        )

    return list(history)
