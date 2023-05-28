import logging
import heapq
from pathlib import Path
from typing import Union

import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from gym.wrappers.monitoring.video_recorder import ImageEncoder

logger = logging.getLogger(__name__)


class VecVideoRecorder(VecEnvWrapper):
    def __init__(
        self,
        venv: VecEnv,
        topk: int = 5,
    ):
        VecEnvWrapper.__init__(self, venv)
        self.frame_buffers = [[] for _ in range(self.num_envs)]
        self.best_videos = []
        self.topk = topk

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.venv.step_wait()
        frames = self.venv.get_images()
        for i, env, done, rew, frame in zip(range(self.num_envs), self.venv.envs, dones, rews, frames):
            buffer = self.frame_buffers[i]
            buffer.append(frame)
            if done:
                if len(self.best_videos) == self.topk:
                    rew, _ = heapq.heappushpop(self.best_videos, (np.sum(rew), buffer))
                    logger.info(f"Dropped video with reward {rew}")
                else:
                    heapq.heappush(self.best_videos, (np.sum(rew), buffer))
                self.frame_buffers[i] = []
                new_obs = env.reset()
                obs[i] = new_obs
        return obs, rews, dones, infos

    def save_videos(self, save_path: Union[str, Path], fps: int, output_fps: int) -> None:
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True, parents=True)
        for i, (_, buffer) in enumerate(self.best_videos):
            enc = ImageEncoder(
                str(save_path / f"video_{i}.mp4"),
                frame_shape=buffer[0].shape,
                frames_per_sec=fps,
                output_frames_per_sec=output_fps,
            )
            for frame in buffer:
                enc.capture_frame(frame)
            enc.close()
