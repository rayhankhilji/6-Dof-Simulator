"""sixdof: 6-DOF reusable rocket flight simulator and landing GNC research system.

Frames: inertial I = flat-Earth ENU at the pad (x East, y North, z Up;
Earth rotation neglected). Body B: x_B longitudinal axis (engine to nose),
y_B / z_B lateral. Attitude quaternion is scalar-first, Hamilton, and maps
body to inertial: v_I = R(q) v_B.
"""

__version__ = "0.1.0"

from .actuators import ActuatorSuite, GimbalActuator, ThrottleActuator
from .control_types import ControlCommand, GuidanceCommand
from .dynamics import RigidBodyDynamics, integrate_rk4, integrate_step, touchdown_check
from .environment import USStandardAtmosphere1976, WindModel, gravity_inertial
from .navigation import EKF, UKF, Navigator, NavState
from .sensors import Barometer, GPS, IMU, Measurements, RadarAltimeter, SensorSuite
from .simulation import NullController, NullGuidance, PerfectNavigator, SimResult, Simulation
from .state import IM, IQ, IR, IV, IW, STATE_SIZE, State, initial_state
from .vehicle import (
    AeroConfig,
    EngineConfig,
    StageConfig,
    Vehicle,
    VehicleConfig,
    falcon9_like_first_stage,
    small_landing_vehicle,
    two_stage_launcher,
)

__all__ = [
    "ActuatorSuite",
    "GimbalActuator",
    "ThrottleActuator",
    "ControlCommand",
    "GuidanceCommand",
    "EKF",
    "UKF",
    "Navigator",
    "NavState",
    "IMU",
    "GPS",
    "Barometer",
    "RadarAltimeter",
    "SensorSuite",
    "Measurements",
    "Simulation",
    "SimResult",
    "NullGuidance",
    "NullController",
    "PerfectNavigator",
    "RigidBodyDynamics",
    "integrate_rk4",
    "integrate_step",
    "touchdown_check",
    "USStandardAtmosphere1976",
    "WindModel",
    "gravity_inertial",
    "State",
    "initial_state",
    "IR",
    "IV",
    "IQ",
    "IW",
    "IM",
    "STATE_SIZE",
    "AeroConfig",
    "EngineConfig",
    "StageConfig",
    "Vehicle",
    "VehicleConfig",
    "falcon9_like_first_stage",
    "small_landing_vehicle",
    "two_stage_launcher",
]
