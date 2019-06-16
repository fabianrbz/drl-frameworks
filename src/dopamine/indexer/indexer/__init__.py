from gym.envs.registration import register

register(
    id='indexer-v0',
    entry_point='indexer.envs:IndexerEnv',
)
