# -*- coding:utf-8 -*-

class Config(object):
    def __init__(self):
        # experiment settings
        self.no_of_simulations = 100
        self.raw_seed = 567

        # environment configuration
        self.true_mean = {"high": 2.0, "low": 1.5}
        self.total_episode = 200  # episodes per game
        self.steps = 100  # steps per episode
        self.states = {'S0': 0, 'S1': 1}
        self.actions = {'A0': 0, 'A1': 1}

        # general agents configuration
        self.alpha_lr = 0.1  # learning rate
        self.gamma_dr = 0.9  # discount rate
        self.stop_explore = 10  # stop explore episodes

        # distributional agent configuration
        self.efficient_estimation_method = None  # method = 'sample_average' or 'EM_MLE'
        self.df = 1.1  # for empirical SP500,  3.29265
        self.mu = None  # (optional for EM_MLE) meta parameters for empirical reward distribution / old: 0.0424105
        self.variance = None  # (optional for EM_MLE) meta parameters for empirical reward distribution

        # categorical agent configuration
        self.v_max = 30
        self.v_min = -30
        self.no_of_atoms = 100  # N
