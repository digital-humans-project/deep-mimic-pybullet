import argparse
import json
import logging

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from model.envs.video_recoder import VecVideoRecorder
from model.multi_clip.selection_policy import SelectionPolicy

logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
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
    eval_env = make_vec_env(env_id, n_envs=1, seed=seed, env_kwargs=env_kwargs, vec_env_cls=DummyVecEnv)

    eval_env = VecVideoRecorder(
        eval_env,
        topk=export_params["topk"],
    )

    # =============
    # Load pre-trained model
    # =============
    model_files = export_params["model_files"]
    models = [PPO.load(model_file, eval_env) for model_file in model_files]
    policies = [model.policy for model in models]
    comp_policy = SelectionPolicy(policies).to(models[0].device)

    # =============
    # start playing
    # =============
    episodes = export_params["max_steps"]

    obs = eval_env.reset()
    for ep in tqdm(range(episodes)):
        action, _ = comp_policy.predict(obs, deterministic=False)
        obs, reward, done, info = eval_env.step(action)
        if done[0]:
            comp_policy.reset()
        eval_env.render()

    eval_env.close()

    export_dir = export_params["out_dir"]
    fps = export_params["fps"]
    eval_env.save_videos(export_dir, export_params["input_fps"], fps)
