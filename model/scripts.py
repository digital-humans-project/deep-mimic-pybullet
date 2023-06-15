import datetime
import os
import pathlib
from shutil import copy as sh_copy

import torch.nn
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common import utils
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
    VecVideoRecorder,
)

from model.algorithms import CustomActorCriticPolicy, CustomPPO
from model.callbacks import (
    CheckpointSaveCallback,
    RecordEvalCallback,
    TensorboardCallback,
)


def manage_save_path(log_dir, name):
    date = datetime.date.today().strftime("%Y-%m-%d-")
    name = date + name
    latest_run_id = utils.get_latest_run_id(log_dir, name)
    save_path = utils.os.path.join(log_dir, f"{name}_{latest_run_id + 1}")
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    return save_path


def train(
    params,
    log_path,
    dir_name,
    debug,
    video_recorder,
    wandb_log,
    config_path="config.json",
):
    """create a model and train it"""
    save_path = manage_save_path(log_path, dir_name)

    # =============
    # unpack params
    # =============

    env_id = params["env_id"]
    hyp_params = params["train_hyp_params"]
    model_params = params["model_params"]
    env_params = params["env_params"]

    n_envs = hyp_params["num_envs"] if (not debug) else 1
    max_episode_steps = hyp_params.get("max_episode_steps", 500)
    max_evaluation_steps = hyp_params.get("max_evaluation_steps", 500)
    seed = hyp_params.get("seed", 313)

    # =============
    # weights and biases
    # =============

    if wandb_log:
        wandb.init(
            project="DH-Project",
            entity="dh-project",
            name=save_path.split("/")[-1],
            config=params,
            sync_tensorboard=True,
            monitor_gym=video_recorder,
            dir=log_path,
        )
        wandb.save(config_path)

    # =============
    # create vectorized environment for training
    # =============

    env = make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_params,
        vec_env_cls=DummyVecEnv if debug else SubprocVecEnv,
    )

    if hyp_params["normalize_observation"] or hyp_params["normalize_reward"]:
        env = VecNormalize(
            env,
            training=True,
            norm_obs=hyp_params["normalize_observation"],
            norm_reward=hyp_params["normalize_reward"],
        )

    # =============
    # create a single environment for evaluation (and video recording)
    # =============

    eval_env = make_vec_env(
        env_id,
        n_envs=1,
        seed=seed,
        env_kwargs={**env_params, 'test_mode': True},
        vec_env_cls=DummyVecEnv,
    )

    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=save_path,
        record_video_trigger=lambda x: x == 0,
        video_length=max_evaluation_steps,
        name_prefix=f"eval-video-{env_id}",
    )

    # =============
    # call backs
    # =============

    # check point: store a trained model
    checkpoint_save_callback = CheckpointSaveCallback(
        save_freq=hyp_params["save_freq"],
        save_path=save_path,
        save_vecnormalize=hyp_params["normalize_observation"] or hyp_params["normalize_reward"],
        export_policy=True,
    )

    # tensorboard: log reward terms
    tensorboard_callback = TensorboardCallback()

    # evaluation w or w/o loaded vecnormalize wrapper: record video
    video_callback = RecordEvalCallback(
        eval_env,
        eval_freq=hyp_params["save_freq"],
        load_path=save_path,
        deterministic=True,
        render=False,
        normalize=(hyp_params["normalize_observation"] or hyp_params["normalize_reward"]),
        video_folder=save_path,
        record_video_trigger=lambda x: x == 0,
        video_length=max_evaluation_steps,
        name_prefix=f"eval-video-{env_id}",
    )

    callbacks = [checkpoint_save_callback]
    if not debug:
        callbacks.append(tensorboard_callback)
    if video_recorder:
        callbacks.append(video_callback)

    # =============
    # create model
    # =============

    activation_fn = torch.nn.Tanh
    if model_params["activation_fn"] == "ReLu":
        activation_fn = torch.nn.ReLU
    elif model_params["activation_fn"] == "TanH":
        activation_fn = torch.nn.Tanh
    elif model_params["activation_fn"] == "Swish":
        activation_fn = torch.nn.SiLU
    elif model_params["activation_fn"] == "ELU":
        activation_fn = torch.nn.ELU

    policy_kwargs = dict(
        activation_fn=activation_fn,
        log_std_init=model_params["log_std_init"],
        net_arch=[
            dict(
                pi=model_params["network_architecture"]["pi"],
                vf=model_params["network_architecture"]["vf"],
            )
        ],
    )

    model = CustomPPO(
        CustomActorCriticPolicy,
        env,
        learning_rate=hyp_params["learning_rate"],
        batch_size=hyp_params["batch_size"],
        n_epochs=hyp_params["n_epochs"],
        n_steps=hyp_params["n_steps"],
        vf_coef=hyp_params["vf_coef"],
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        seed=seed,
    )
    # model = PPO.load("log/2023-05-15-PylocoVanilla-v0-bob_humanoid_reward-45.0M_2/model_25600000_steps.zip",env)

    logging_format = ["stdout"]
    if not debug:
        logging_format.append("tensorboard")
    my_logger = configure(save_path, logging_format)
    model.set_logger(my_logger)

    # =============
    # save config file and reward file
    # =============

    sh_copy(config_path, os.path.join(save_path, "config.json"))

    # =============
    # start training
    # =============

    model.learn(total_timesteps=hyp_params["time_steps"], callback=callbacks)
    model.save(save_path + "/model")

    del model
