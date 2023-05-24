import argparse
import json
import os.path
import sys

import matplotlib

matplotlib.use("Agg")

from pylocogym.cmake_variables import *

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
    parser.add_argument("--logDir", help="Name of log directory to use for prediction", required=False)
    parser.add_argument("--step", help="Predict using the model after n time steps of training", required=False)
    parser.add_argument("-wb", "--wandb", help="Enable logging to wandb", required=False, action="store_true")

    # do not pass args to sub-functions for a better readability.
    args = parser.parse_args()

    # log path
    log_path = PYLOCO_LOG_PATH
    data_path = PYLOCO_DATA_PATH

    if not args.test:
        # =============
        # training
        # =============

        # config file
        if args.config is None:
            sys.exit("Config name needs to be specified for training: --config <config file name>")
        else:
            config_path = os.path.join(data_path, "conf", args.config)
            print("- config file path = {}".format(config_path))

        with open(config_path, "r") as f:
            params = json.load(f)

        # loading file
        urdf_file = "data/robots/deep-mimic/humanoid.urdf"
        motion_clip_file = "humanoid3d_walk.txt"
        motion_clip_file = os.path.join("data", "deepmimic", "motions", motion_clip_file)

        # train parameters
        hyp_params = params["train_hyp_params"]
        steps = hyp_params["time_steps"]
        dir_name = "{id}-{clips}-{steps:.1f}M".format(
            id=params["env_id"], clips=motion_clip_file, steps=float(steps / 1e6)
        )

        # training
        scripts.train(
            params=params,
            log_path=log_path,
            dir_name=dir_name,
            debug=args.debug,
            video_recorder=args.videoRecorder,
            wandb_log=args.wandb,
            config_path=config_path,
            motion_clips_path=motion_clip_file,
            urdf_path=urdf_file,
        )
