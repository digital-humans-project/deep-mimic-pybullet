from typing import List

import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from torch import Tensor, nn


class SelectionPolicy(BasePolicy):
    def __init__(
        self,
        policies: List[ActorCriticPolicy],
        tempreture=0.3,
        alpha=1,
        mode="weighted",
        phase_idx=0,
    ):
        super().__init__(observation_space=policies[0].observation_space, action_space=policies[0].action_space)
        self.policies = nn.ModuleList(policies)
        self.tempreture = tempreture
        self.values = nn.Parameter(torch.ones(len(policies)))
        self.alpha = alpha
        self.mode = mode
        self.policy_seq = [2, 3, 2, 3]
        self.cur_policy = 0
        self.cur_policy_indices = [0] * len(policies)
        self.phase_idx = phase_idx
        self.phases = [0] * len(policies)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def reset(self):
        self.cur_policy = 0
        self.cur_policy_indices = [0] * len(self.policies)
        self.phases = [0] * len(self.policies)

    def _predict(self, observation: Tensor, deterministic: bool = False) -> Tensor:
        res = []
        for i in range(len(observation)):
            phase = observation[i, self.phase_idx]
            print("phase", phase)
            if phase * self.phases[i] < 0:
                self.cur_policy_indices[i] = np.clip(self.cur_policy_indices[i] + 1, 0, len(self.policy_seq) - 1)

            action = self.policies[self.policy_seq[self.cur_policy_indices[i]]]._predict(
                observation[[i]], deterministic
            )
            self.phases[i] = phase
            res.append(action)
        res = torch.cat(res)
        return res
