try:
    from itertools import pairwise  # type: ignore
except ImportError:
    from itertools import tee

    def pairwise(iterable):
        # pairwise('ABCDEFG') --> AB BC CD DE EF FG
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


from typing import Optional

from model.data.dataset import IterableKeyframeMotionDataset, MotionDataSample


class ContinuousMotionDataset:
    """
    `ContinuousMotionDataset` is the base class for all continuous motion datasets.

    Continuous motion datasets are datasets that can be evaluated at any time `t` and
    return a `MotionDataSample` that contains the state of the motion at time `t`.
    """

    def __init__(self, kf_dataset: IterableKeyframeMotionDataset) -> None:
        self.kf_dataset = kf_dataset

    def reset(self) -> None:
        """
        Reset the dataset to the beginning.
        """
        raise NotImplementedError

    def eval(self, t: float) -> Optional[MotionDataSample]:
        """
        Evaluate the motion at time `t`.

        If `t` is outside the range of the dataset, `None` is returned.
        """
        raise NotImplementedError


def lerp(a, b, alpha):
    return a * (1 - alpha) + b * alpha


class LerpMotionDataset(ContinuousMotionDataset):
    """
    `LerpMotionDataset` is a wrapper around `IterableKeyframeMotionDataset` that
    interpolates between keyframes using linear interpolation.
    """

    def __init__(self, kf_dataset: IterableKeyframeMotionDataset) -> None:
        super().__init__(kf_dataset)
        self.kf_dataset = kf_dataset
        self.reset()

    def reset(self) -> None:
        self.kf_iter = iter(self.kf_dataset)
        self.cur_kf = next(self.kf_iter)

    def eval(self, t: float) -> Optional[MotionDataSample]:
        assert (
            t >= self.cur_kf.t0
        ), f"t = {t} < time of current keyframe {self.cur_kf.t0}, t must be monotonically increasing"
        while t > self.cur_kf.t1:
            try:
                self.cur_kf = next(self.kf_iter)
            except StopIteration:
                return None
        kf = self.cur_kf
        alpha = (t - kf.t0) / kf.dt
        q = lerp(kf.q0, kf.q1, alpha)
        phase = lerp(kf.phase0, kf.phase1, alpha)
        return self.kf_dataset.SampleType.BaseSampleType(t, q, kf.qdot, phase)
