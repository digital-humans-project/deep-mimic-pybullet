from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ClassVar, Dict, Iterator, Tuple, Type, Union

import numpy as np
from numpy.typing import NDArray


class StrEnum(str, Enum):
    """
    Enum class interchangable with strings.
    """

    def __new__(cls, value, *args, **kwargs):
        if not isinstance(value, (str, auto)):
            raise TypeError(f"Values of StrEnums must be strings: {value!r} is a {type(value)}")
        return super().__new__(cls, value, *args, **kwargs)

    def __str__(self) -> str:
        return str(self.value)

    def _generate_next_value_(name, *_):
        return name.lower()


class Fields:
    """
    Utility class for accessing fields of `q` and `qdot` vectors.

    Fields access is done by attribute access, e.g. `q_fields.root_pos` or indexing, e.g. `q_fields['root_pos']`.

    The names of the fields are defined in the `FieldNames` enum class.
    """

    def __init__(self, data: NDArray) -> None:
        self.data = data

    FieldNames: ClassVar[Type[Enum]] = StrEnum
    fields: ClassVar[Dict[FieldNames, Tuple[int, int]]] = {}

    def __getattr__(self, __name: str) -> NDArray:
        return self[__name]

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in self.fields:
            self[__name] = __value
        else:
            super().__setattr__(__name, __value)

    def __getitem__(self, __name: Union[FieldNames, str]) -> NDArray:
        range = self.fields[self.FieldNames(__name)]
        return self.data[range[0] : range[1]]

    def __setitem__(self, __name: Union[FieldNames, str], value: NDArray) -> None:
        range = self.fields[self.FieldNames(__name)]
        self.data[range[0] : range[1]] = value


@dataclass
class MotionDataSample:
    """
    Data sample for a single time step of a motion.
    """

    t: float
    """
    Time of the sample.
    """

    q: np.ndarray
    """
    Configuration vector.
    """

    qdot: np.ndarray
    """
    First derivative of the configuration vector.
    """

    phase: float = 0
    """
    Phase of the motion in the range [0, 1].
    """

    FieldsType: ClassVar[Type[Fields]] = Fields
    """
    Type of the `q_fields` and `qdot_fields` properties. Used for accessing fields of `q` and `qdot` vectors.
    """

    @property
    def q_fields(self) -> FieldsType:
        """
        Accessor for the `q` vector fields.
        """
        return self.FieldsType(self.q)

    @property
    def qdot_fields(self) -> FieldsType:
        """
        Accessor for the `qdot` vector fields.
        """
        return self.FieldsType(self.qdot)


@dataclass
class KeyframeMotionDataSample:
    """
    Data sample for a keyframe of a motion in a time interval.
    """

    t0: float
    """
    Start time of the keyframe.
    """

    q0: np.ndarray
    """
    Configuration vector at the start of the keyframe.
    """

    q1: np.ndarray
    """ 
    Configuration vector at the end of the keyframe.
    """

    qdot: np.ndarray
    """
    First derivative of the configuration vector, defined by `qdot = (q1 - q0) / dt`.
    """

    dt: float
    """
    Duration of the keyframe.
    """

    phase0: float = 0
    """
    Phase of the motion at the start of the keyframe in the range [0, 1].
    """

    phase1: float = 0
    """
    Phase of the motion at the end of the keyframe.
    """

    FieldsType: ClassVar[Type[Fields]] = Fields
    BaseSampleType: ClassVar[Type[MotionDataSample]] = MotionDataSample

    @property
    def q0_fields(self) -> FieldsType:
        """
        Accessor for the `q0` vector fields.
        """
        return self.FieldsType(self.q0)

    @property
    def q1_fields(self) -> FieldsType:
        """
        Accessor for the `q1` vector fields.
        """
        return self.FieldsType(self.q1)

    @property
    def qdot_fields(self) -> FieldsType:
        """
        Accessor for the `qdot` vector fields.
        """
        return self.FieldsType(self.qdot)

    @property
    def t1(self) -> float:
        """
        End time of the keyframe.
        """
        return self.t0 + self.dt


class IterableKeyframeMotionDataset:
    """
    Base class for iterable keyframe motion datasets.
    """

    SampleType: Type[KeyframeMotionDataSample] = KeyframeMotionDataSample

    def __init__(self) -> None:
        pass

    def __iter__(self) -> Iterator[SampleType]:
        raise NotImplementedError


class MapKeyframeMotionDataset(IterableKeyframeMotionDataset):
    """
    Base class for keyframe motion datasets that can be indexed.
    """

    SampleType = KeyframeMotionDataSample

    def __init__(self) -> None:
        pass

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> SampleType:
        raise NotImplementedError

    def __iter__(self) -> Iterator[SampleType]:
        for i in range(len(self)):
            yield self[i]

    @property
    def duration(self) -> float:
        """
        Duration of the motion dataset.
        """
        return self[-1].t1 - self[0].t0
