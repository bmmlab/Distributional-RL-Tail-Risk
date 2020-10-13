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
from copy import deepcopy
import scipy.stats as sts
import pandas as pd
import json

from outlier_project.utils.config import *
from outlier_project.utils.policies import *
from outlier_project.utils.utils import *
from outlier_project.envs import leptokurtosis


def efficient_estimates(reward_list, mu=None, sigma_square=None, df=None, method=None):
    """
    Efficient estimation of the reward_list (distribution)
    :param mu: mean can be provided otherwise calculated using sample average
    :param sigma_square: variance can be provided otherwise will be calculated
    :param reward_list: reward history for a particular state-action pair
    :param df: degree of freedom, for student t distribution
    :param method: the way to calculate estimates of the mean
    :return: efficient estimate of the mean
    """

    if method == 'sample_average':
        return np.average(reward_list)

    elif method == 'EM_MLE':  # Expectation Maximization algorithm
        if sigma_square is None:
            sigma_square = np.var(reward_list)

        if mu is None:
            mu = np.average(reward_list)

        x1 = (df + 1) * sigma_square
        x2 = df * sigma_square + np.square(np.array(reward_list) - mu)
        weights = np.divide(x1, x2)
        estimates = np.sum(weights * reward_list) / np.sum(weights)
        return estimates

    elif method == 'exponential':
        fitting_para = sts.expon.fit(-np.array(reward_list)[1:])
        return -fitting_para[0]


def distributional_sarsa(env, c):
    alpha_lr = c.alpha_lr  # learning rate
    gamma_dr = c.gamma_dr  # discount rate
    stop_explore = c.stop_explore  # stop explore episodes

    states = c.states
    actions = c.actions
    accumulated_reward = dict(zip(list(states.keys()), [0] * len(list(states.keys()))))

    states_keys = list(states.keys())
    actions_keys = list(actions.keys())

    # initialize q table for distributional SARSA
    q_table = pd.DataFrame(data=0.0, index=actions_keys, columns=states_keys)

    # initialize reward memory for distributional SARSA
    reward_dist = initialize_distribution(states=states_keys, actions=actions_keys)

    history = deque([])  # record data for statistical analysis
    reward_list = deque()

    for each_episode in range(0, c.total_episode):
        next_state = env.state

        for each_step in range(0, c.steps):
            current_state = next_state  # update the current states

            # core distributional SARSA
            action_key = e_greedy(state=current_state, action_list=actions_keys,
                                  q_table=q_table, episode=each_episode,
                                  stop_explore=stop_explore)

            next_state, reward, done, _ = env.step(action=actions[action_key])

            reward_dist[current_state][action_key].append(reward)  # add reward to reward list

            reward_estimates = efficient_estimates(reward_list=reward_dist[current_state][action_key],
                                                   mu=c.mu,
                                                   sigma_square=c.variance,
                                                   df=c.df,
                                                   method=c.efficient_estimation_method)

            q_value = q_table[current_state][action_key]
            q_value_next = q_table[next_state].max()
            td_delta = reward_estimates + gamma_dr * q_value_next - q_value
            q_value += alpha_lr * td_delta

            q_table.at[action_key, current_state] = q_value  # update q value

            reward_list.append(reward)
            accumulated_reward[current_state] += reward

        # record important history per episode
        history.append(
            {
                "episode": each_episode,
                "current_state": current_state,
                "current_action": action_key,
                "td_delta": td_delta,
                "q_table": q_table.copy().to_dict(),
                "accumulated_reward": accumulated_reward.copy(),
                "game_reward": np.sum(reward_list),
                # "reward_distribution": deepcopy(reward_dist)
            }
        )

    return list(history)
