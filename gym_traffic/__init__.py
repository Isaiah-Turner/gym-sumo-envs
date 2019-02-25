from gym.envs.registration import register

register(
    id='Traffic-tIntersection-cli-v0',
    entry_point='gym_traffic.envs:TrafficEnvSimple',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "cli", "network": "T-Intersection", "prefix": "tIntersection"},
    nondeterministic=True
)

register(
    id='Traffic-tIntersection-gui-v0',
    entry_point='gym_traffic.envs:TrafficEnvSimple',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "gui", "network": "T-Intersection", "prefix": "tIntersection"},
    nondeterministic=True
)

register(
    id='Traffic-Simple-gui-v0',
    entry_point='gym_traffic.envs:TrafficEnvSimple',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "gui"},
    nondeterministic=True
)

register(
    id='Traffic-Simple-cli-v0',
    entry_point='gym_traffic.envs:TrafficEnvSimple',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "cli"},
    nondeterministic=True
)


