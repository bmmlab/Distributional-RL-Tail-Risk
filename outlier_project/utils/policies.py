# -*- coding:utf-8 -*-
import numpy as np


def e_greedy(state, action_list, q_table, episode=0, stop_explore=10):
    """
    policy $\pi$, e-greedy policy
    x% that the agent will choose action randomly
    after certain episode threshold, the random search stops
    greedy policy: agent choose action with max value
    :param state: current state
    :param q_table: q values
    :param episode: current episode
    :param stop_explore: episode that exploration completely stops
    :return: action choice
    """

    if episode < stop_explore:
        # exploration = 0.9 * np.exp(-0.1 * episode)
        exploration = 0.9 * np.exp(-0.001 * episode)
        random_draw = np.random.rand()

        if random_draw <= exploration:
            return np.random.choice(action_list)

        else:
            return q_table[state].idxmax()

    else:
        return q_table[state].idxmax()
