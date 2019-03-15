from gym.envs.registration import register
# DC Medium network
register(
    id='Traffic-DCMed-cli-v0',
    entry_point='gym_traffic.envs:TrafficEnvSimple',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "cli", "network": "DCMedNetwork", "prefix": "DCMed"},
    nondeterministic=True
)
register(
    id='Traffic-DCMed-gui-v0',
    entry_point='gym_traffic.envs:TrafficEnvSimple',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "gui", "network": "DCMedNetwork", "prefix": "DCMed"},
    nondeterministic=True
)
# DC 2way
register(
    id='Traffic-2way-cli-v0',
    entry_point='gym_traffic.envs:TrafficEnvSimple',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "cli", "network": "DC2WayIntersection", "prefix": "2way"},
    nondeterministic=True
)
register(
    id='Traffic-2way-gui-v0',
    entry_point='gym_traffic.envs:TrafficEnvSimple',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "gui", "network": "DC2WayIntersection", "prefix": "2way"},
    nondeterministic=True
)
# yIntersection
register(
    id='Traffic-litteRiver-cli-v0',
    entry_point='gym_traffic.envs:TrafficEnvSimple',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "cli", "network": "LittleRiver4WayIntersection", "prefix": "littleriver"},
    nondeterministic=True
)
register(
    id='Traffic-litteRiver-gui-v0',
    entry_point='gym_traffic.envs:TrafficEnvSimple',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "gui", "network": "LittleRiver4WayIntersection", "prefix": "littleriver"},
    nondeterministic=True
)
# yIntersection
register(
    id='Traffic-yIntersection-cli-v0',
    entry_point='gym_traffic.envs:TrafficEnvSimple',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "cli", "network": "Y-Intersection", "prefix": "yIntersection"},
    nondeterministic=True
)
register(
    id='Traffic-yIntersection-gui-v0',
    entry_point='gym_traffic.envs:TrafficEnvSimple',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "gui", "network": "Y-Intersection", "prefix": "yIntersection"},
    nondeterministic=True
)
# tIntersection
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
#simple two way
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


