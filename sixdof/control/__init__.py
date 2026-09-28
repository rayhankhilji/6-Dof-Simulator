"""TVC/RCS controllers."""

from .lqr import LQRController
from .mpc import LinearMPCController
from .nonlinear import GeometricController
from .pid import PIDAttitudeController

CONTROLLERS = {
    "pid": PIDAttitudeController,
    "lqr": LQRController,
    "nonlinear": GeometricController,
    "mpc": LinearMPCController,
}


def make_controller(name: str, vehicle=None):
    """Instantiate a controller by name."""
    if name not in CONTROLLERS:
        raise ValueError(f"unknown controller {name!r}; options: {sorted(CONTROLLERS)}")
    return CONTROLLERS[name]()


__all__ = [
    "PIDAttitudeController",
    "LQRController",
    "GeometricController",
    "LinearMPCController",
    "CONTROLLERS",
    "make_controller",
]
