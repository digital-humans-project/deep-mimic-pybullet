import argparse
import json
import logging

import cv2 as cv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from model.algorithms.custom_ppo import CustomPPO
from model.envs.video_recoder import VecVideoRecorder

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-l", "--loop", default="none")
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
    action_rescale = env_kwargs["rescale_actions"]
    time_per_frame = 1/export_params["input_fps"]
    loop_type = args.loop

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
        action_rescale=action_rescale,
        time_per_frame=time_per_frame,
        loop_type=loop_type,
    )

    model = CustomPPO.load(model_file, eval_env, device="cuda")

    obs = eval_env.reset()
    for ep in tqdm(range(export_params["max_steps"])):
        action, _ = model.predict(obs)
        #print(action)
        #print(action.shape)
        obs, reward, done, info = eval_env.step(action)
        #print("done", done)
        #print("reward", reward)
        #print("info", info)
        #cv.imshow("frame", info[0]["frame"][:, :, ::-1])
        #cv.waitKey(1)
    # input_fps = round(1.0 / env_kwargs["time_step"])

    eval_env.close()

    export_dir = export_params["out_dir"]
    fps = export_params["fps"]
    eval_env.save_videos(export_dir, export_params["input_fps"], fps)
