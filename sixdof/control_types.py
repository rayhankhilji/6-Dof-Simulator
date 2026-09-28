"""Shared control/guidance command types (used by controllers in Phase 3+)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ControlCommand:
    """Actuator-level command: throttle in [0,1], gimbal angles [rad],
    and an RCS torque request in the body frame [N m]."""

    throttle: float = 0.0
    gimbal_y: float = 0.0
    gimbal_z: float = 0.0
    rcs_torque_B: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class GuidanceCommand:
    """Guidance-level command produced by the guidance block.

    Attributes
    ----------
    accel_cmd_I : (3,) array
        Desired inertial acceleration of the CG *excluding* gravity
        (i.e. the thrust acceleration) [m/s^2].
    throttle : float
        Commanded throttle hint in [0,1].
    engine_on : bool
        Whether the main engine should be burning.
    q_des : (4,) array or None
        Desired attitude quaternion (body->inertial), if guidance provides one.
    phase : str
        Guidance phase label (e.g. "coast", "powered_descent", "terminal").
    r_ref, v_ref : (3,) arrays or None
        Reference position/velocity for tracking controllers.
    t_go : float
        Time-to-go estimate [s].
    """

    accel_cmd_I: np.ndarray = field(default_factory=lambda: np.zeros(3))
    throttle: float = 0.0
    engine_on: bool = False
    q_des: Optional[np.ndarray] = None
    phase: str = "idle"
    r_ref: Optional[np.ndarray] = None
    v_ref: Optional[np.ndarray] = None
    t_go: float = 0.0
