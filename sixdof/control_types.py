"""Shared control/guidance command types (used by controllers in Phase 3+)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ControlCommand:
    """Actuator-level command: throttle in [0,1], gimbal angles [rad]."""

    throttle: float = 0.0
    gimbal_y: float = 0.0
    gimbal_z: float = 0.0


@dataclass
class GuidanceCommand:
    """Guidance-level command produced by the guidance block.

    Attributes
    ----------
    accel_cmd_I : (3,) array
        Commanded inertial acceleration [m/s^2].
    thrust_dir_des_I : (3,) array or None
        Desired thrust direction in the inertial frame (alternative to
        ``q_des``).
    q_des : (4,) array or None
        Desired attitude quaternion (body->inertial).
    throttle : float
        Commanded throttle [0,1] feed-forward.
    phase : str
        Guidance phase label (e.g. "boost", "descent", "landing").
    """

    accel_cmd_I: np.ndarray = None
    thrust_dir_des_I: Optional[np.ndarray] = None
    q_des: Optional[np.ndarray] = None
    throttle: float = 0.0
    phase: str = "idle"
