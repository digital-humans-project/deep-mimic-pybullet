import argparse
import json
import os.path
import sys

import matplotlib

matplotlib.use("Agg")

from . import scripts

if __name__ == "__main__":
    # example of python script for training and testing
    parser = argparse.ArgumentParser(description="Process input arguments.")
    parser.add_argument("-t", "--test", help="Test (prediction) mode.", required=False, action="store_true")
    parser.add_argument("-d", "--debug", help="Debug mode.", required=False, action="store_true")
    parser.add_argument(
        "-vr",
        "--videoRecorder",
        help="Activate video recorder to record a video footage.",
        required=False,
        action="store_true",
    )
    parser.add_argument("-c", "--config", help="Path to config file", required=False)
    # parser.add_argument('--rewardFile', help="Path to reward file", required=True)
    parser.add_argument("--logDir", help="Name of log directory to use for prediction", default="logs")
    parser.add_argument("--step", help="Predict using the model after n time steps of training", required=False)
    parser.add_argument("-wb", "--wandb", help="Enable logging to wandb", required=False, action="store_true")

    # do not pass args to sub-functions for a better readability.
    args = parser.parse_args()

    exp_name = "deepmimic-pybullet"

    if not args.test:
        # =============
        # training
        # =============

        # config file
        if args.config is None:
            sys.exit("Config name needs to be specified for training: --config <config file name>")
        else:
            config_path = args.config
            print("- config file path = {}".format(config_path))

        with open(config_path, "r") as f:
            params = json.load(f)

        # train parameters
        hyp_params = params["train_hyp_params"]
        steps = hyp_params["time_steps"]
        env_params = params["env_params"]
        arg_file = env_params.get("arg_file", "run_humanoid3d_unknown_args.txt")
        motion_name = arg_file.lstrip("run_humanoid3d_").rstrip("_args.txt")

        dir_name = "{id}-{rew}-{steps:.1f}M".format(id=params["env_id"], rew=motion_name, steps=float(steps / 1e6))

        # training
        scripts.train(
            params=params,
            log_path=args.logDir,
            dir_name=dir_name,
            debug=args.debug,
            video_recorder=args.videoRecorder,
            wandb_log=args.wandb,
            config_path=config_path,
        )
