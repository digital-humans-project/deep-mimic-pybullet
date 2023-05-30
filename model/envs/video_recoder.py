import heapq
import logging
from pathlib import Path
from typing import Optional, Union

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
    ):
        VecEnvWrapper.__init__(self, venv)
        self.frame_buffers = [[] for _ in range(self.num_envs)]
        self.total_rews = np.zeros(self.num_envs)
        self.best_videos = []
        self.topk = topk

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        return obs

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
        if len(self.best_videos) == self.topk:
            if total_rew > self.best_videos[0][0]:
                r, l, _ = heapq.heappushpop(self.best_videos, (total_rew, length, buffer))
                logger.info(f"Dropped video with reward = {r}, length = {l}")
        else:
            heapq.heappush(self.best_videos, (total_rew, length, buffer))
        logger.info(f"New video with reward {total_rew}, length = {length}")

    def save_videos(self, save_path: Union[str, Path], fps: int, output_fps: int) -> None:
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True, parents=True)
        for i, (rew, l, buffer) in enumerate(self.best_videos):
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
