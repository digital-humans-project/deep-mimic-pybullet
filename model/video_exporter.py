import argparse
import json
import logging

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm
import cv2 as cv
from model.envs.video_recoder import VecVideoRecorder

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        params = json.load(f)

    export_params = params["export"]
    model_file = export_params["model_file"]
    env_id = params["env_id"]
    hyp_params = params["train_hyp_params"]

    max_episode_steps = hyp_params.get("max_episode_steps", 5000)
    seed = hyp_params.get("seed", 313)
    env_kwargs = params["env_params"]

    eval_env = make_vec_env(
        env_id,
        n_envs=hyp_params["num_envs"],
        seed=seed,
        env_kwargs=env_kwargs,
        vec_env_cls=DummyVecEnv,
    )
    eval_env = VecVideoRecorder(
        eval_env,
        topk=export_params["topk"],
    )

    model = PPO.load(model_file, eval_env, device="cuda")

    obs = eval_env.reset()
    for ep in tqdm(range(export_params["max_steps"])):
        action, _ = model.predict(obs)
        obs, reward, done, info = eval_env.step(action)
        # cv.imshow("frame", info[0]["frame"][:, :, ::-1])
        # cv.waitKey(1)
    # input_fps = round(1.0 / env_kwargs["time_step"])

    eval_env.close()

    export_dir = export_params["out_dir"]
    fps = export_params["fps"]
    eval_env.save_videos(export_dir, export_params["input_fps"], fps)
