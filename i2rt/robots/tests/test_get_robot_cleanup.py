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


def test_real_robot_accepts_gravity_idle_tuning_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers can tune a leader without changing the shared arm-family YAML."""
    chain = FakeMotorChain()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(factory, "DMChainCanInterface", lambda *_args, **_kwargs: chain)

    def build_robot(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(factory, "MotorChainRobot", build_robot)
    gravity = np.array([1.0, 1.1, 1.1, 1.2, 1.0, 1.0])
    damping = np.array([0.07, 0.07, 0.07, 0.18, 0.04, 0.04])
    friction = np.array([0.30, 0.30, 0.30, 0.06, 0.06, 0.06])

    factory.get_yam_robot(
        gripper_type=GripperType.YAM_TEACHING_HANDLE,
        gravity_comp_factor=gravity,
        grav_comp_kd=damping,
        coulomb_friction=friction,
    )

    np.testing.assert_allclose(captured["gravity_comp_factor"], gravity)
    np.testing.assert_allclose(captured["grav_comp_kd"], damping)
    np.testing.assert_allclose(captured["coulomb_friction"], friction)
    assert captured["use_coulomb_friction"] is False
