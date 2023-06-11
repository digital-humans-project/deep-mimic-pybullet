import gym
from gym.envs.registration import make, registry, spec


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


# ------------bullet-------------

register(
    id="HumanoidDeepMimicBackflipBulletEnv-v1",
    entry_point="pybullet_envs.deep_mimic.gym_env:HumanoidDeepMimicBackflipBulletEnv",
    max_episode_steps=2000,
    reward_threshold=2000.0,
)

register(
    id="HumanoidDeepMimicWalkBulletEnv-v1",
    entry_point="pybullet_envs.deep_mimic.gym_env:HumanoidDeepMimicWalkBulletEnv",
    max_episode_steps=2000,
    reward_threshold=2000.0,
)

register(
    id="HumanoidDeepMimicBulletEnv-v1",
    entry_point="pybullet_envs.deep_mimic.gym_env:HumanoidDeepBulletEnv",
    max_episode_steps=2000,
    reward_threshold=2000.0,
)
