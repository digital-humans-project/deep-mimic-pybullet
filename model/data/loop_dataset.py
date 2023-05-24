from typing import List

from model.data.dataset import IterableKeyframeMotionDataset


class LoopKeyframeMotionDataset(IterableKeyframeMotionDataset):
    """
    `LoopKeyframeMotionDataset` is a wrapper around `IterableKeyframeMotionDataset` that
    loops over the dataset `num_loop` times. If `num_loop` is -1, then the dataset will
    loop forever.

    It also tracks the state of the keyframe fields specified in `track_fields` and
    add the offset of the last keyframe in the previous loop to the tracked fields.
    """

    def __init__(
        self,
        ds: IterableKeyframeMotionDataset,
        num_loop=-1,
        track_fields: List[str] = [],
    ) -> None:
        super().__init__()
        self.SampleType = ds.SampleType
        self.ds = ds
        self.num_loop = num_loop
        self.track_fields = track_fields

    def __iter__(self):
        loop = 0
        last_state = {k: 0 for k in self.track_fields}  # tracked fields
        t = 0
        while self.num_loop < 0 or loop < self.num_loop:
            first_kf = None
            last_kf = None  # track last keyframe
            for i, kf in enumerate(self.ds):
                kf.t0 += t
                for k, v in last_state.items():
                    kf.q0_fields[k] += v
                    kf.q1_fields[k] += v
                if i == 0:
                    first_kf = kf
                last_kf = kf
                yield kf
            if last_kf is not None:
                # update tracked fields state to be the end of the last keyframe
                t = last_kf.t1
                for k in self.track_fields:
                    last_state[k] = last_kf.q1_fields[k] - first_kf.q0_fields[k]  # type: ignore
            loop += 1
