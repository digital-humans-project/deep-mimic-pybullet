import heapq
import logging
from pathlib import Path
from typing import Optional, Union
import json

import numpy as np
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
    VecEnvWrapper,
)

logger = logging.getLogger(__name__)


class VecVideoRecorder(VecEnvWrapper):
    def __init__(
        self,
        venv: VecEnv,
        topk: int = 5,
        action_rescale: bool = True,
        time_per_frame: float = 1/30,
        loop_type: str = "none"
    ):
        VecEnvWrapper.__init__(self, venv)
        self.frame_buffers = [[] for _ in range(self.num_envs)]
        self.action_buffers = [[] for _ in range(self.num_envs)]
        self.total_rews = np.zeros(self.num_envs)
        self.best_videos = []
        self.topk = topk
        self.action_rescale = action_rescale
        self.time_per_frame = time_per_frame
        self.loop_type = loop_type

        out_offset = np.array([
        0.0000000000, 0.0000000000, 0.0000000000, -0.200000000, 0.0000000000, 0.0000000000,
        0.0000000000, -0.200000000, 0.0000000000, 0.0000000000, 0.00000000, -0.2000000, 1.57000000,
        0.00000000, 0.00000000, 0.00000000, -0.2000000, 0.00000000, 0.00000000, 0.00000000,
        -0.2000000, -1.5700000, 0.00000000, 0.00000000, 0.00000000, -0.2000000, 1.57000000,
        0.00000000, 0.00000000, 0.00000000, -0.2000000, 0.00000000, 0.00000000, 0.00000000,
        -0.2000000, -1.5700000
        ])

        out_scale = np.array([
        0.20833333333333, 1.00000000000000, 1.00000000000000, 1.00000000000000, 0.25000000000000,
        1.00000000000000, 1.00000000000000, 1.00000000000000, 0.12077294685990, 1.00000000000000,
        1.000000000000, 1.000000000000, 0.159235668789, 0.159235668789, 1.000000000000,
        1.000000000000, 1.000000000000, 0.079617834394, 1.000000000000, 1.000000000000,
        1.000000000000, 0.159235668789, 0.120772946859, 1.000000000000, 1.000000000000,
        1.000000000000, 0.159235668789, 0.159235668789, 1.000000000000, 1.000000000000,
        1.000000000000, 0.107758620689, 1.000000000000, 1.000000000000, 1.000000000000,
        0.159235668789
        ])

        self.action_mean = -out_offset
        self.action_std = 1.0 / out_scale

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        return obs
    
    def step_async(self, actions):
        if self.action_rescale:
                actions_rectified = (actions.copy() - self.action_mean) / self.action_std
        time_and_root = [self.time_per_frame, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        for i in range(self.num_envs):
            curr_action = self.action_buffers[i]
            curr_action.append(time_and_root+actions_rectified[i].tolist())
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.venv.step_wait()
        self.total_rews += rews
        frames = self.venv.get_images()
        for i, env, done, rew, frame, info in zip(range(self.num_envs), self.venv.envs, dones, rews, frames, infos):
            buffer = self.frame_buffers[i]
            buffer.append(frame)
            info["frame"] = frame
            if done:
                self.push_video(i)
                self.frame_buffers[i] = []
                self.action_buffers[i] = []
                new_obs = env.reset()
                self.total_rews[i] = 0
                obs[i] = new_obs
        return obs, rews, dones, infos

    def close(self) -> None:
        for i in range(self.num_envs):
            if len(self.frame_buffers[i]) > 0:
                self.push_video(i)
        super().close()

    def push_video(self, i):
        total_rew = self.total_rews[i]
        buffer = self.frame_buffers[i]
        length = len(buffer)
        curr_action = self.action_buffers[i] 

        if len(self.best_videos) == self.topk:
            if total_rew > self.best_videos[0][0]:
                r, l, _, _ = heapq.heappushpop(self.best_videos, (total_rew, length, buffer, curr_action))
                logger.info(f"Dropped video with reward = {r}, length = {l}")
        else:
            heapq.heappush(self.best_videos, (total_rew, length, buffer, curr_action))
        logger.info(f"New video with reward {total_rew}, length = {length}")

    def save_videos(self, save_path: Union[str, Path], fps: int, output_fps: int) -> None:
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True, parents=True)
        for i, (rew, l, buffer, curr_action) in enumerate(self.best_videos):
            
            save_action_path = str(save_path/f"action_{i}_rew_{rew:02f}_{l/fps:.01f}s.txt")
            action = curr_action.copy()
            #action = action.tolist()
            data = {
                "Loop": self.loop_type,
                "Frames": action
                }
            with open(save_action_path, 'w') as f:
                json.dump(data, f)

            enc = ImageEncoder(
                str(save_path / f"video_{i}_rew_{rew:02f}_{l/fps:.01f}s.mp4"),
                frame_shape=buffer[0].shape,
                frames_per_sec=fps,
                output_frames_per_sec=output_fps,
            )
            for frame in buffer:
                enc.capture_frame(frame)
            enc.close()

    def render(self, mode: str = "human"):
        VecEnv.render(self, mode=mode)
