# -*- coding:utf-8 -*-
from gym.envs.registration import register

register(
    id='Leptokurtosis-v0',
    entry_point='outlier_project.envs.leptokurtosis:LeptokurticEnv',
)
