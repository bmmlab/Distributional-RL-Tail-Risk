# -*- coding:utf-8 -*-
#######################################################################
# Copyright (C) 2018 -                                                #
# Shijie Huang (harveyh@student.unimelb.edu.au)                       #
# Brain, Mind and Markets Laboratory                                  #
# https://research.unimelb.edu.au/brain-mind-markets                  #
# University of Melbourne                                             #
#######################################################################


import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd


class LeptokurticEnv(gym.Env):
    """
    Two states, Two actions environment

    One has to specify which reward distribution to use once the class has initialized
    if self.reward_distribution_type == 'G', then Gaussian reward distribution will be used
    No degree of freedom is required

    if self.reward_distribution_type == 'T', then Student-t reward distribution will be used
    self.df needs to be initialized

    if self.reward_distribution_type == 'E', then empirical distribution will be used
    self.empirical_distribution needs to be initialized, one should pass a list or an array of
    empirical data

    Two states: 0 and 1, the probability of state transition is 70-30, that is
    70% chance the next state will be the same, 30% chance the next state will be different

    Agents' actions: 0 and 1, this is passed to self.step(action) each step
    agent will get a reward, and the reward is randomly drawn from one of the three specified
    reward distribution

    optimal policy: (state 0, action 0) or (state 1, action 1)
    """

    def __init__(self):
        self.action_space = spaces.Discrete(2)  # two actions
        self.observation_space = spaces.Discrete(2)  # two states
        self.reward_distribution_type = None
        self.reward_distribution = None  # reward distribution
        self.cum_reward = 0.0
        self.df = 1.1

        self.true_mean = None

        self.seed()
        self.viewer = None
        self.state = "S0"

        self.steps_beyond_done = None

    def seed(self, seed=None):
        np.random.seed(seed)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if self.reward_distribution_type == 'E':
            f = np.random.choice(self.reward_distribution)
        elif self.reward_distribution_type == 'T':
            f = self.reward_distribution(df=self.df)
        else:
            f = self.reward_distribution()

        if self.state == "S0":
            if action == 0:
                reward = f + self.true_mean["high"]
            else:
                reward = f + self.true_mean["low"]
        elif self.state == "S1":
            if action == 0:
                reward = f + self.true_mean["low"]
            else:
                reward = f + self.true_mean["high"]

        if self.state == "S0":
            self.state = np.random.choice(["S0", "S1"], p=[0.7, 0.3])
        elif self.state == "S1":
            self.state = np.random.choice(["S0", "S1"], p=[0.3, 0.7])

        self.cum_reward += reward

        done = False
        done = bool(done)

        return self.state, reward, done, {}

    def initialize_reward_distribution(self):
        """
        You may choose to call this function to
        generate reward distribution
        :return:
        """
        if self.reward_distribution_type == 'G':
            self.reward_distribution = np.random.standard_normal
        elif self.reward_distribution_type == 'T':
            self.reward_distribution = np.random.standard_t
        elif self.reward_distribution_type == 'E':
            self.reward_distribution = pd.read_csv("../utils/sp500.csv")["daily_return"].to_numpy()
        else:
            raise NotImplementedError

    def reset(self):
        self.state = self.state = np.random.randint(0, 2, dtype=np.int)
        self.steps_beyond_done = None
        return self.state

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
