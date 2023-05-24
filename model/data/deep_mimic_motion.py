import json
from dataclasses import dataclass
from enum import auto
from pathlib import Path
from typing import ClassVar, Dict, Literal, Optional, Tuple, Union

import numpy as np
from pylocogym.data.dataset import (
    Fields,
    KeyframeMotionDataSample,
    MapKeyframeMotionDataset,
    MotionDataSample,
    StrEnum,
)


class DeepMimicMotionDataFieldNames(StrEnum):
    """
    Enum class for DeepMimic motion data field names.
    """

    ROOT_POS = auto()
    ROOT_ROT = auto()
    CHEST_ROT = auto()
    NECK_ROT = auto()
    R_HIP_ROT = auto()
    R_KNEE_ROT = auto()
    R_ANKLE_ROT = auto()
    R_SHOULDER_ROT = auto()
    R_ELBOW_ROT = auto()
    L_HIP_ROT = auto()
    L_KNEE_ROT = auto()
    L_ANKLE_ROT = auto()
    L_SHOULDER_ROT = auto()
    L_ELBOW_ROT = auto()


class DeepMimicMotionDataField(Fields):
    FieldNames = DeepMimicMotionDataFieldNames
    fields: Dict[DeepMimicMotionDataFieldNames, Tuple[int, int]] = {
        DeepMimicMotionDataFieldNames.ROOT_POS: (0, 3),
        DeepMimicMotionDataFieldNames.ROOT_ROT: (3, 7),
        DeepMimicMotionDataFieldNames.CHEST_ROT: (7, 11),
        DeepMimicMotionDataFieldNames.NECK_ROT: (11, 15),
        DeepMimicMotionDataFieldNames.R_HIP_ROT: (15, 19),
        DeepMimicMotionDataFieldNames.R_KNEE_ROT: (19, 20),
        DeepMimicMotionDataFieldNames.R_ANKLE_ROT: (20, 24),
        DeepMimicMotionDataFieldNames.R_SHOULDER_ROT: (24, 28),
        DeepMimicMotionDataFieldNames.R_ELBOW_ROT: (28, 29),
        DeepMimicMotionDataFieldNames.L_HIP_ROT: (29, 33),
        DeepMimicMotionDataFieldNames.L_KNEE_ROT: (33, 34),
        DeepMimicMotionDataFieldNames.L_ANKLE_ROT: (34, 38),
        DeepMimicMotionDataFieldNames.L_SHOULDER_ROT: (38, 42),
        DeepMimicMotionDataFieldNames.L_ELBOW_ROT: (42, 43),
    }


@dataclass
class DeepMimicMotionDataSample(MotionDataSample):
    FieldsType: ClassVar = DeepMimicMotionDataField


@dataclass
class DeepMimicKeyframeMotionDataSample(KeyframeMotionDataSample):
    FieldsType: ClassVar = DeepMimicMotionDataField
    BaseSampleType: ClassVar = DeepMimicMotionDataSample


class DeepMimicMotion(MapKeyframeMotionDataset):
    """
    DeepMimic motion data.

    It contains a single motion clip.
    """

    SampleType = DeepMimicKeyframeMotionDataSample

    def __init__(
        self,
        path: Union[str, Path],
        t0: float = 0.0,
        loop: Optional[Literal["wrap", "none", "mirror"]] = None,
    ) -> None:
        super().__init__()

        with open(path, "r") as f:
            data = json.load(f)

        self.loop = data["Loop"] if loop is None else loop
        assert self.loop in ["wrap", "none", "mirror"]

        frames = np.array(data["Frames"])
        if self.loop == "mirror":
            frames = np.concatenate([frames, frames[-2::-1]])

        self.dt = frames[:, 0]
        t = np.cumsum(self.dt)
        self.t = np.concatenate([[0], t])[:-1]
        self.q = frames[:, 1:]
        self.qdot = np.diff(self.q, axis=0) / self.dt[:-1, None]
        self.t0 = t0

    def __len__(self) -> int:
        # dataset length is the number of keyframes (intervals) = number of frames - 1
        return len(self.qdot)

    @property
    def duration(self) -> float:
        return self.t[-1]

    def __getitem__(self, idx) -> DeepMimicKeyframeMotionDataSample:
        idx = range(len(self))[idx]
        t = self.t[idx].item()
        return DeepMimicKeyframeMotionDataSample(
            dt=self.dt[idx].item(),
            t0=t + self.t0,
            q0=self.q[idx, :].copy(),
            q1=self.q[idx + 1, :].copy(),
            qdot=self.qdot[idx, :].copy(),
            phase0=t / self.duration,
            phase1=self.t[idx + 1].item() / self.duration,
        )
