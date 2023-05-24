import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
from pylocogym.data.dataset import IterableKeyframeMotionDataset
from pylocogym.data.deep_mimic_motion import (
    DeepMimicKeyframeMotionDataSample,
    DeepMimicMotion,
)


class DeepMimicMotionDataset(IterableKeyframeMotionDataset):
    SampleType = DeepMimicKeyframeMotionDataSample

    def __init__(
        self,
        dataset_path: Union[Path, str],
        data_dir: Optional[Union[Path, str]] = None,
    ) -> None:
        dataset_path = Path(dataset_path).resolve()
        if data_dir is None:
            data_dir = dataset_path / "../.."
        self.data_dir = Path(data_dir).resolve()

        with open(dataset_path) as f:
            data = json.load(f)
            motions = data["Motions"]

        self.weights = np.array([m["Weight"] for m in motions]).astype(np.float32)
        self.weights /= self.weights.sum()
        self.motions = [self.data_dir / Path(*Path(m["File"]).parts[1:]) for m in motions]
        self.t = 0.0

    def __iter__(self):
        n = len(self.motions)
        while True:
            motion = np.random.choice(n, p=self.weights)
            for sample in DeepMimicMotion(self.motions[motion], self.t):
                yield sample
                self.t = sample.t
