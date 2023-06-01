from typing import List

import torch
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from torch import Tensor, nn


class ComposeActorCriticPolicy(BasePolicy):
    def __init__(
        self,
        policies: List[ActorCriticPolicy],
        tempreture=0.3,
        alpha=1,
        mode="weighted",
    ):
        super().__init__(observation_space=policies[0].observation_space, action_space=policies[0].action_space)
        self.policies = nn.ModuleList(policies)
        self.tempreture = tempreture
        self.values = nn.Parameter(torch.ones(len(policies)))
        self.alpha = alpha
        self.mode = mode

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def _predict(self, observation: Tensor, deterministic: bool = False) -> Tensor:
        values = [p.predict_values(observation).squeeze() for p in self.policies]
        # actions = [
        #     p.get_distribution(observation).get_actions(deterministic=deterministic).squeeze() for p in self.policies
        # ]
        actions = [p._predict(observation, deterministic) for p in self.policies]
        values = torch.stack(values)
        actions = torch.stack(actions)
        values = values * self.alpha + self.values * (1 - self.alpha)
        self.values.data = values

        if self.mode == "weighted":
            weights = (values / self.tempreture).softmax(dim=0)
            action = torch.einsum("pia,p->ia", actions, weights)
        elif self.mode == "max":
            imax = torch.argmax(values)
            print(imax)
            action = actions[imax]
        else:
            raise NotImplementedError
        return action
