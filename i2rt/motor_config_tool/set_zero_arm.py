"""Zero all joint motors of one arm in a single run.

Hold (or rest) the arm at its zero pose — all six joint coordinates zero, i.e.
the folded rest pose — then run:

    python i2rt/motor_config_tool/set_zero_arm.py --channel can_follower_l

The current position of every motor is shown first and nothing is written to
motor flash until you confirm (skip the prompt with --yes). Power cycle the arm
afterwards so the driver re-anchors on the new offsets.

The gripper (motor 7) is excluded by default because its zero reference is the
jaws-fully-closed position, not the arm zero pose. To zero it in the same run,
close the jaws fully and pass --include-gripper.

Note: the 400 ms motor watchdog can fire between the enable and save frames,
which makes a save attempt report a disabled motor — attempts are retried
(--retries) to ride through this.
"""

import logging
import sys

import tyro

from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface
from i2rt.robots.utils import ArmType, _load_arm_config


def _read_position(interface: DMSingleMotorCanInterface, motor_id: int, motor_type: str) -> float:
    """Enable the motor and read its current position with a zero-gain MIT frame."""
    interface.motor_on(motor_id, motor_type)
    return interface.set_control(motor_id, motor_type, 0, 0, 0, 0, 0).position


def main(
    channel: str,
    arm: ArmType = ArmType.YAM,
    include_gripper: bool = False,
    gripper_motor_type: str = "DM4310",
    retries: int = 3,
    tolerance: float = 0.05,
    yes: bool = False,
) -> None:
    """Save the current pose as the zero position for every motor of one arm.

    Args:
        channel: CAN interface the arm is on (e.g. can_follower_l).
        arm: Arm variant; determines the joint motor IDs and types.
        include_gripper: Also zero the gripper motor (ID 7). Only do this with the jaws fully closed.
        gripper_motor_type: Motor type of the gripper (DM4310 for linear/crank 4310, DM3507 for linear 3507).
        retries: Save attempts per motor before giving up (the 400 ms watchdog can eat a frame).
        tolerance: Max |position| (rad) accepted as a successful zero after saving.
        yes: Skip the confirmation prompt.
    """
    logging.basicConfig(level=logging.WARNING)

    motors = [(can_id, motor_type) for can_id, motor_type in _load_arm_config(arm).motor_list]
    if include_gripper:
        motors.append((0x07, gripper_motor_type))

    interface = DMSingleMotorCanInterface(channel=channel, bustype="socketcan", control_mode=ControlMode.MIT)
    try:
        print(f"Reading current positions on {channel} ({arm.value}):")
        positions: dict[int, float] = {}
        for motor_id, motor_type in motors:
            try:
                positions[motor_id] = _read_position(interface, motor_id, motor_type)
                print(f"  motor {motor_id} ({motor_type}): {positions[motor_id]:+.4f} rad")
            except Exception as e:  # noqa: BLE001
                print(f"  motor {motor_id} ({motor_type}): UNREACHABLE ({e})")
            finally:
                interface.motor_off(motor_id)

        missing = [motor_id for motor_id, _ in motors if motor_id not in positions]
        if missing:
            print(f"Aborting: motors {missing} did not respond. Check CAN link state and stale processes.")
            sys.exit(1)

        if not yes:
            answer = input("Save these positions as zero? This writes motor flash. [y/N] ")
            if answer.strip().lower() not in ("y", "yes"):
                print("Aborted, nothing written.")
                sys.exit(1)

        failed: list[int] = []
        for motor_id, motor_type in motors:
            ok = False
            for attempt in range(1, retries + 1):
                try:
                    interface.motor_on(motor_id, motor_type)
                    interface.save_zero_position(motor_id)
                    position = interface.set_control(motor_id, motor_type, 0, 0, 0, 0, 0).position
                    if abs(position) < tolerance:
                        print(f"  motor {motor_id}: zeroed (now {position:+.4f} rad)")
                        ok = True
                        break
                    print(f"  motor {motor_id}: still at {position:+.4f} rad after save (attempt {attempt}/{retries})")
                except Exception as e:  # noqa: BLE001
                    print(f"  motor {motor_id}: attempt {attempt}/{retries} failed ({e})")
                finally:
                    interface.motor_off(motor_id)
            if not ok:
                failed.append(motor_id)

        if failed:
            print(f"FAILED to zero motors: {failed}. Re-run for these IDs (watchdog may have eaten a frame).")
            sys.exit(1)
        print("All motors zeroed. Power cycle the arm before running anything.")
    finally:
        interface.close()


if __name__ == "__main__":
    tyro.cli(main)
