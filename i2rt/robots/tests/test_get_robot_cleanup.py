"""Construction failures stop the already-enabled real-hardware CAN command thread."""

from types import SimpleNamespace
from typing import Any, NoReturn

import numpy as np
import pytest

from i2rt.robots import get_robot as factory
from i2rt.robots.utils import GripperType


class FakeMotorChain:
    def __init__(self):
        self.motor_offset = np.zeros(6)
        self.started = False
        self.closed = False

    def read_states(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(pos=0.0) for _ in range(6)]

    def start_thread(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True


def test_real_robot_construction_failure_closes_the_enabled_motor_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    chain = FakeMotorChain()

    def build_chain(*_args: Any, **_kwargs: Any) -> FakeMotorChain:
        return chain

    monkeypatch.setattr(factory, "DMChainCanInterface", build_chain)

    def fail_robot(**_kwargs: Any) -> NoReturn:
        raise RuntimeError("bad model after motor enable")

    monkeypatch.setattr(factory, "MotorChainRobot", fail_robot)

    with pytest.raises(RuntimeError, match="bad model"):
        factory.get_yam_robot(gripper_type=GripperType.YAM_TEACHING_HANDLE)

    assert chain.started
    assert chain.closed
