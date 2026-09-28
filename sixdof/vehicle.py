"""Vehicle configuration and runtime mass-properties bookkeeping.

Body frame B: x_B is the longitudinal axis (engine/base at x=0, nose at
x=+length), y_B and z_B lateral. All positions along the axis are measured
from the base of the stage.

Mass model per stage: a solid cylinder of dry structure spanning the stage
length, plus a propellant cylinder that fills the stage length when full and
shrinks toward the base as it depletes (propellant height = fill fraction *
length, occupying [0, h_prop]). The propellant CG therefore moves toward the
base as propellant burns. Inertia of each component is a solid-cylinder
tensor referenced to the current total CG via the parallel-axis theorem.

Governing relations:

    m_total = sum(stage dry + prop remaining) + payload
    x_cg    = sum(m_i x_i) / m_total
    J_x     = 0.5 m r^2                         (cylinder about axis)
    J_y=J_z = m (3 r^2 + h^2)/12 + m dx^2       (parallel axis)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

G0 = 9.80665


@dataclass
class AeroConfig:
    """Aerodynamic coefficient configuration for a stage.

    Attributes
    ----------
    ref_area : float
        Reference area [m^2].
    cd_table : tuple of (mach, cd) arrays
        Axial drag coefficient vs Mach.
    cn_alpha : float
        Normal-force slope per radian of angle of attack.
    cm_damping : float
        Pitch/yaw damping coefficient (nondimensional rate w*L/(2V)).
    cl_roll_damping : float
        Roll damping coefficient (nondimensional rate p*L/(2V)).
    """

    ref_area: float
    cd_table: tuple = (
        np.array([0.0, 0.8, 1.0, 1.1, 1.3, 2.0, 5.0]),
        np.array([0.30, 0.32, 0.45, 0.60, 0.55, 0.40, 0.25]),
    )
    cn_alpha: float = 2.0
    cm_damping: float = -0.3
    cl_roll_damping: float = -0.05

    def __post_init__(self) -> None:
        self.cd_table = tuple(np.asarray(t, dtype=float) for t in self.cd_table)

    def cd(self, mach: float) -> float:
        """Interpolated axial drag coefficient at ``mach``."""
        return float(np.interp(mach, self.cd_table[0], self.cd_table[1]))


@dataclass
class EngineConfig:
    """Engine model with back-pressure loss, Isp interpolation, actuator lags.

    Sea-level thrust for a vacuum-rated engine:

        T(p) = T_vac_cmd - p_amb * A_e * n_engines,   clamped >= 0

    Isp is linear in ambient pressure between isp_vac (p=0) and isp_sl
    (p=101325 Pa). Mass flow uses vacuum values: mdot = T_vac_cmd/(Isp_vac g0).

    Attributes
    ----------
    thrust_max_vac : float
        Total max vacuum thrust of all engines [N].
    thrust_min_frac : float
        Minimum throttleable fraction of max thrust.
    isp_vac, isp_sl : float
        Vacuum and sea-level specific impulse [s].
    nozzle_exit_area : float
        Per-engine nozzle exit area [m^2].
    gimbal_max : float
        Max gimbal deflection per axis [rad].
    gimbal_rate_max : float
        Max gimbal rate [rad/s].
    throttle_tau, gimbal_tau : float
        First-order actuator time constants [s].
    n_engines : int
        Number of engines (all deflect together).
    """

    thrust_max_vac: float
    thrust_min_frac: float = 0.4
    isp_vac: float = 311.0
    isp_sl: float = 282.0
    nozzle_exit_area: float = 1.0
    gimbal_max: float = np.radians(8.0)
    gimbal_rate_max: float = np.radians(20.0)
    throttle_tau: float = 0.1
    gimbal_tau: float = 0.05
    n_engines: int = 1

    def thrust_and_mdot(self, throttle: float, p_amb: float) -> tuple[float, float]:
        """Actual thrust [N] and mass flow [kg/s] at ``throttle``, ``p_amb``.

        ``throttle`` in [0, 1]: 0 means engine off; nonzero values are clamped
        to the throttleable floor ``thrust_min_frac``.
        """
        throttle = float(throttle)
        if throttle <= 0.0:
            return 0.0, 0.0
        throttle = min(max(throttle, self.thrust_min_frac), 1.0)
        t_vac_cmd = throttle * self.thrust_max_vac
        thrust = max(t_vac_cmd - p_amb * self.nozzle_exit_area * self.n_engines, 0.0)
        mdot = t_vac_cmd / (self.isp_vac * G0)
        return thrust, mdot

    def isp(self, p_amb: float) -> float:
        """Isp [s] interpolated linearly in ambient pressure."""
        frac = np.clip(p_amb / 101325.0, 0.0, 1.0)
        return float(self.isp_vac + (self.isp_sl - self.isp_vac) * frac)


@dataclass
class StageConfig:
    """Geometry, mass and aero configuration for one stage.

    Attributes
    ----------
    dry_mass, prop_mass : float
        Dry and (full) propellant mass [kg].
    length, diameter : float
        Stage length and diameter [m].
    engine : EngineConfig
    cg_offset_from_base_dry : float
        Dry-structure CG along x_B from the base [m].
    cg_offset_prop : float
        Full-propellant CG along x_B from the base [m]. Used as the fill
        fraction reference: prop occupies [0, 2*cg_offset_prop] when full.
    gimbal_point : float
        x_B location of the engine pivot [m]; 0 = base.
    cp_offset : float
        Center of pressure along x_B from the base [m].
    aero : AeroConfig
    """

    dry_mass: float
    prop_mass: float
    length: float
    diameter: float
    engine: EngineConfig
    cg_offset_from_base_dry: float
    cg_offset_prop: float
    gimbal_point: float
    cp_offset: float
    aero: AeroConfig


@dataclass
class VehicleConfig:
    """Stacked vehicle: ``stages[0]`` is the bottom (first) stage.

    Stages are stacked along x_B: stage i occupies
    [x_stack_i, x_stack_i + length_i] where x_stack_i is the cumulative length
    of the stages below. Payload sits at the top.
    """

    stages: List[StageConfig]
    payload_mass: float = 0.0

    def stage_base(self, stage_index: int) -> float:
        """x_B position of the base of ``stage_index`` in the stacked vehicle [m]."""
        return sum(s.length for s in self.stages[:stage_index])

    def _dry_mass_at_or_below(self, stage_index: int) -> float:
        """Dry mass of active stage plus all stages/payload above it [kg]."""
        m = self.payload_mass
        for j in range(stage_index, len(self.stages)):
            s = self.stages[j]
            m += s.dry_mass + (0.0 if j == stage_index else s.prop_mass)
        return m

    def stages_above_mass(self, stage_index: int, prop_remaining: float) -> float:
        """Mass [kg] of all stages above ``stage_index`` plus payload."""
        m = self.payload_mass
        for j in range(stage_index + 1, len(self.stages)):
            s = self.stages[j]
            prop = s.prop_mass if j > stage_index else 0.0
            m += s.dry_mass + prop
        return m

    def _components(self, stage_index: int, prop_remaining: float):
        """Yield (mass, cg_abs, radius, cyl_height) for each component.

        cg_abs is the CG measured in body x_B from the vehicle base.
        Components: for each live stage j >= stage_index a dry cylinder and a
        propellant cylinder (full prop for upper stages); payload as a point
        mass at the top.
        """
        comps = []
        for j in range(stage_index, len(self.stages)):
            s = self.stages[j]
            base = self.stage_base(j)
            r = s.diameter / 2.0
            comps.append((s.dry_mass, base + s.cg_offset_from_base_dry, r, s.length))
            m_prop = prop_remaining if j == stage_index else s.prop_mass
            if m_prop > 0.0:
                fill = m_prop / s.prop_mass
                h_prop = fill * 2.0 * s.cg_offset_prop
                cg_prop = base + h_prop / 2.0
                comps.append((m_prop, cg_prop, r, max(h_prop, 1e-6)))
        if self.payload_mass > 0.0:
            top = sum(s.length for s in self.stages)
            comps.append((self.payload_mass, top, 0.0, 0.0))
        return comps

    def total_mass(self, stage_index: int, prop_remaining: float) -> float:
        """Total vehicle mass [kg] at the given stack state."""
        return sum(c[0] for c in self._components(stage_index, prop_remaining))

    def cg_position(self, stage_index: int, prop_remaining: float) -> float:
        """CG along x_B from the vehicle base [m]."""
        comps = self._components(stage_index, prop_remaining)
        m = sum(c[0] for c in comps)
        return sum(c[0] * c[1] for c in comps) / m

    def inertia_tensor(self, stage_index: int, prop_remaining: float) -> np.ndarray:
        """3x3 inertia tensor about the current CG, body frame [kg m^2]."""
        comps = self._components(stage_index, prop_remaining)
        cg = self.cg_position(stage_index, prop_remaining)
        J = np.zeros((3, 3))
        for m, cg_i, r, h in comps:
            Jx = 0.5 * m * r * r
            Jy = m * (3.0 * r * r + h * h) / 12.0 + m * (cg_i - cg) ** 2
            J += np.diag([Jx, Jy, Jy])
        return J


class Vehicle:
    """Runtime vehicle state: active stage and remaining propellant.

    Attributes
    ----------
    config : VehicleConfig
    active_stage : int
        Index of the current (bottom-most live) stage.
    prop_remaining : float
        Propellant mass remaining in the active stage [kg].
    events : list
        Log of staging events (dicts).
    """

    def __init__(self, config: VehicleConfig, active_stage: int = 0) -> None:
        self.config = config
        self.active_stage = active_stage
        self.prop_remaining = config.stages[active_stage].prop_mass
        self.events: list[dict] = []

    @property
    def active_stage_config(self) -> StageConfig:
        """Active stage configuration."""
        return self.config.stages[self.active_stage]

    def burn(self, m_dot: float, dt: float) -> float:
        """Consume ``m_dot*dt`` of propellant; return actual mass burned."""
        dm = min(max(m_dot * dt, 0.0), self.prop_remaining)
        self.prop_remaining -= dm
        return dm

    def stage(self, t: float = 0.0) -> None:
        """Jettison the active stage and activate the next one."""
        if self.active_stage + 1 >= len(self.config.stages):
            raise RuntimeError("no upper stage to activate")
        dropped = self.active_stage_config
        self.events.append(
            {"t": t, "event": "stage_sep", "dropped_dry_mass": dropped.dry_mass}
        )
        self.active_stage += 1
        self.prop_remaining = self.active_stage_config.prop_mass

    def mass_properties(
        self, total_mass: float | None = None
    ) -> tuple[float, float, np.ndarray]:
        """Return ``(m_total, cg_x, J)`` for the current stack state.

        If ``total_mass`` is given (e.g. the integrated state mass), the
        propellant remaining is derived as
        ``clip(total_mass - dry_stack_mass, 0, prop_mass)`` so CG and inertia
        stay consistent with the integrated mass during propagation.
        """
        if total_mass is None:
            prop = self.prop_remaining
            m = self.config.total_mass(self.active_stage, prop)
        else:
            dry = self.config._dry_mass_at_or_below(self.active_stage)
            prop = float(np.clip(total_mass - dry, 0.0, self.active_stage_config.prop_mass))
            m = float(total_mass)
        cg = self.config.cg_position(self.active_stage, prop)
        J = self.config.inertia_tensor(self.active_stage, prop)
        return m, cg, J


# ---------------------------------------------------------------------------
# Example vehicles (approximate public figures, clearly not flight data).
# ---------------------------------------------------------------------------


def falcon9_like_first_stage() -> StageConfig:
    """Approximate Falcon-9-like first stage / landing booster config.

    Rough public numbers: ~27 t dry, ~30 t landing propellant reserve,
    one Merlin-class engine 845 kN vac, Isp 311/282 s, 8 deg gimbal,
    35% throttle floor, 41 m x 3.7 m.
    """
    engine = EngineConfig(
        thrust_max_vac=845e3,
        thrust_min_frac=0.35,
        isp_vac=311.0,
        isp_sl=282.0,
        nozzle_exit_area=1.6,
        gimbal_max=np.radians(8.0),
        gimbal_rate_max=np.radians(20.0),
        n_engines=1,
    )
    aero = AeroConfig(ref_area=np.pi * (3.7 / 2.0) ** 2)
    return StageConfig(
        dry_mass=27_000.0,
        prop_mass=30_000.0,
        length=41.0,
        diameter=3.7,
        engine=engine,
        cg_offset_from_base_dry=8.0,  # engine-heavy base section
        cg_offset_prop=10.0,
        gimbal_point=0.0,
        # Descending base-first, the relative wind comes from below (the -x_B
        # direction of travel). Static stability then requires the CP *above*
        # the CG along +x_B (grid fins near the top act like dart feathers).
        cp_offset=11.0,  # ~3 m above the typical descent CG (~8 m)
        aero=aero,
    )


def small_landing_vehicle() -> VehicleConfig:
    """Single-stage landing test vehicle (Falcon-9-like booster alone)."""
    return VehicleConfig(stages=[falcon9_like_first_stage()], payload_mass=0.0)


def two_stage_launcher() -> VehicleConfig:
    """Approximate two-stage launcher (Falcon-9-like, no flight data).

    Stage 1: ~25 t dry, ~400 t prop, 9 Merlin-class engines (845 kN vac each).
    Stage 2: ~4 t dry, ~90 t prop, one Merlin-Vac-class engine 934 kN, Isp 348 s.
    Payload: 10 t.
    """
    s1 = falcon9_like_first_stage()
    s1.prop_mass = 400_000.0
    s1.engine = EngineConfig(
        thrust_max_vac=9 * 845e3,
        thrust_min_frac=0.4,
        isp_vac=311.0,
        isp_sl=282.0,
        nozzle_exit_area=1.6,
        gimbal_max=np.radians(5.0),
        gimbal_rate_max=np.radians(20.0),
        n_engines=9,
    )
    s1.cg_offset_prop = 15.0

    eng2 = EngineConfig(
        thrust_max_vac=934e3,
        thrust_min_frac=0.0,
        isp_vac=348.0,
        isp_sl=348.0,
        nozzle_exit_area=20.0,
        gimbal_max=np.radians(5.0),
        n_engines=1,
    )
    s2 = StageConfig(
        dry_mass=4_000.0,
        prop_mass=90_000.0,
        length=14.0,
        diameter=3.7,
        engine=eng2,
        cg_offset_from_base_dry=6.0,
        cg_offset_prop=5.0,
        gimbal_point=0.0,
        cp_offset=8.0,
        aero=AeroConfig(ref_area=np.pi * (3.7 / 2.0) ** 2),
    )
    return VehicleConfig(stages=[s1, s2], payload_mass=10_000.0)
