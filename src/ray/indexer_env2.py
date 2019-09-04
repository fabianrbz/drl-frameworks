import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import re
import sql_metadata
import os
import psycopg2
import yaml
import re
import pdb
from gym.spaces import Discrete, Box, Dict
from ray.tune.registry import register_env
from indexer_env_or import IndexerEnvOr
from gym.envs.registration import register

SCHEMA = "../dopamine/indexer/indexer/envs/resources/schema.sql"
FOLDER = "../dopamine/indexer/indexer/envs/dummy"
QUERIES_DIRECTORY = f"{FOLDER}/queries/"
BEST_COST = 344568.11
BEST_INDEXES = ['l_partkey', 'l_comment', 'l_shipdate', 'l_discount', 'l_suppkey', 'l_receiptdate', 'l_extendedprice', 'l_tax', 'l_orderkey']
INITIAL = 18099161.35


class IndexerEnv2(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, config):
      self.action_space = Discrete(16)
      self.wrapped = IndexerEnvOr()
      self.observation_space = Dict({
          "avail_actions": Box(0, 1, shape=(16, )),
          "indexer": self.wrapped.observation_space,
          })

  def reset(self):
    self.avail_actions = np.array([1] * self.action_space.n)
    return {
        "avail_actions": self.avail_actions,
        "indexer": self.wrapped.reset(),
    }

  def step(self, action):
    orig_obs, rew, done, info = self.wrapped.step(action)
    self.avail_actions[action] = 0
    obs = {
        "avail_actions": self.avail_actions,
        "indexer": orig_obs,
    }
    return obs, rew, done, info
