import json
import logging
import os.path
import sys
import time

import matplotlib
from tqdm import tqdm

from model.composite_policy import ComposeActorCriticPolicy
from model.envs.video_recoder import VecVideoRecorder

matplotlib.use("Agg")

import cv2 as cv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

import envs  # noqa: F401 pylint: disable=unused-import

logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    motion_clip_file = "humanoid3d_jump.txt"
    config = "conf/pybullet_debug_env.json"

    model_files = [
        "model_data/pybullet/2023-05-29-HumanoidDeepMimicBulletEnv-v1-walk-45.0M_1/model_38000000_steps.zip",
        "model_data/pybullet/2023-05-28-HumanoidDeepMimicBulletEnv-v1-jog-45.0M_3/model_44000000_steps.zip",
        # "model_data/pybullet/2023-05-26-HumanoidDeepMimicBackflipBulletEnv-v1-deepmimic-pybullet-45.0M_1/model_42000000_steps.zip",
        # "model_data/pybullet/2023-05-28-HumanoidDeepMimicBulletEnv-v1-cartwheel-45.0M_3/model_40000000_steps.zip",
        # "/home/howyoung/Code/eth/digital-humans/deep-mimic-pybullet-run/logs/2023-05-30-HumanoidDeepMimicBulletEnv-v1-getup_facedown-45.0M_4/model_43200000_steps.zip",
        # "model_data/pybullet/2023-05-30-HumanoidDeepMimicBulletEnv-v1-dance_a-45.0M_1/model_38000000_steps.zip",
    ]

    with open(config, "r") as f:
        params = json.load(f)

    export_params = params["export"]
    env_id = params["env_id"]
    hyp_params = params["train_hyp_params"]

    max_episode_steps = hyp_params.get("max_episode_steps", 5000)
    seed = hyp_params.get("seed", 313)
    env_kwargs = params["env_params"]

    # =============
    # create a simple environment for testing model
    # =============

    # eval_env = gym.make(env_id, **env_kwargs)
    eval_env = make_vec_env(env_id, n_envs=1, seed=seed, env_kwargs=env_kwargs, vec_env_cls=DummyVecEnv)

    eval_env = VecVideoRecorder(
        eval_env,
        topk=export_params["topk"],
    )

    # =============
    # Load pre-trained model
    # =============
    models = [PPO.load(model_file, eval_env) for model_file in model_files]
    policies = [model.policy for model in models]
    comp_policy = ComposeActorCriticPolicy(policies).to(models[0].device)

    # =============
    # start playing
    # =============
    episodes = 1000
    frame_rate = 60

    # output = cv.VideoWriter("test.mp4", cv.VideoWriter_fourcc(*"MP4V"), 15, (1280, 960))

    obs = eval_env.reset()
    for ep in tqdm(range(episodes)):
        # eval_env.envs[0].render("human")
        # action, _ = comp_policy.predict(obs, deterministic=True)
        action, _ = comp_policy.predict(obs, deterministic=False)
        # action, _ = models[0].policy.predict(obs, deterministic=False)
        # print((action - action1))
        obs, reward, done, info = eval_env.step(action)

        # print("now phase", obs[-1])

    eval_env.close()
    # output.release()
    export_dir = export_params["out_dir"]
    fps = export_params["fps"]
    eval_env.save_videos(export_dir, export_params["input_fps"], fps)
