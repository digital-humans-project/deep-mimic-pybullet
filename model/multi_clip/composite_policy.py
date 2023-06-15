from typing import List

import torch
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from torch import Tensor, nn


class CompositeActorCriticPolicy(BasePolicy):
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
        self.cur_policy = -1
        self.phase_idx = phase_idx
        self.phases = [0] * len(policies)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def _predict(self, observation: Tensor, deterministic: bool = False) -> Tensor:
        res = []
        for i in range(len(observation)):
            phase = observation[i, self.phase_idx]
            print("phase", phase)
            if phase * self.phases[i] < 0 or self.cur_policy == -1:
                values = [p.predict_values(observation[[i]]).squeeze() for p in self.policies]
                actions = [p._predict(observation[[i]], deterministic) for p in self.policies]
                self.cur_policy = torch.argmax(torch.tensor(values)).item()
                print("change policy to", self.cur_policy)
                action = actions[self.cur_policy]
            else:
                action = self.policies[self.cur_policy]._predict(observation[[i]], deterministic)
            self.phases[i] = phase
            res.append(action)
        res = torch.cat(res)
        return res
