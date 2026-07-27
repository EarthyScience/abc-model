from dataclasses import dataclass, replace
from typing import Generic

import jax
import jax.numpy as jnp
from jax import Array

from ..abstracts import (
    AbstractAtmosphereModel,
    AbstractAtmosphereState,
    AbstractCoupledState,
    AbstractLandModel,
    AbstractRadiationModel,
    AbstractState,
    LandT,
    RadT,
)
from ..utils import PhysicalConstants as cst
from .abstracts import (
    AbstractCloudModel,
    AbstractMixedLayerModel,
    AbstractSurfaceLayerModel,
    CloudT,
    SurfT,
)
from .clouds import NoCloudModel
from .dayonly import DayOnlyAtmosphereState
from .mixed_layer.bulk import BulkState
from .residual_layer.residual import ResidualLayerModel, ResidualLayerState
from .stable_layer.sbl import SBLModel, SBLState


@dataclass
class ActiveBLState(AbstractState):
    """Wrapper combining daytime convective mixed layer and nighttime stable boundary layer.

    The ``is_night`` flag determines which sub-state drives the dynamics.
    """

    is_night: Array
    mixed: BulkState
    sbl: SBLState


@dataclass
class DayAndNightAtmosphereState(AbstractAtmosphereState, Generic[SurfT, CloudT]):
    """Atmosphere state with day/night switching.

    During daytime the ``active_bl.mixed`` (convective mixed layer) is active;
    at night ``active_bl.sbl`` (stable boundary layer) takes over.
    The ``residual`` preserves the daytime mixed-layer properties aloft overnight.
    """

    surface: SurfT
    active_bl: ActiveBLState
    residual: ResidualLayerState
    clouds: CloudT

    # same for day and night
    @property
    def ra(self) -> Array:
        return self.surface.ra

    @property
    def thetasurf(self) -> Array:
        return self.surface.thetasurf

    @property
    def ustar(self) -> Array:
        return self.surface.ustar

    @property
    def uw(self) -> Array:
        return self.surface.uw

    @property
    def vw(self) -> Array:
        return self.surface.vw

    # these switch between day and night
    @property
    def is_night(self) -> Array:
        return self.active_bl.is_night

    @property
    def theta(self) -> Array:
        return jnp.where(
            self.active_bl.is_night,
            self.active_bl.sbl.theta,
            self.active_bl.mixed.theta,
        )

    @property
    def q(self) -> Array:
        return jnp.where(
            self.active_bl.is_night, self.active_bl.sbl.q, self.active_bl.mixed.q
        )

    @property
    def co2(self) -> Array:
        return jnp.where(
            self.active_bl.is_night, self.active_bl.sbl.co2, self.active_bl.mixed.co2
        )

    @property
    def surf_pressure(self) -> Array:
        return jnp.where(
            self.active_bl.is_night,
            self.active_bl.sbl.surf_pressure,
            self.active_bl.mixed.surf_pressure,
        )

    @property
    def u(self) -> Array:
        return jnp.where(
            self.active_bl.is_night, self.active_bl.sbl.u, self.active_bl.mixed.u
        )

    @property
    def v(self) -> Array:
        return jnp.where(
            self.active_bl.is_night, self.active_bl.sbl.v, self.active_bl.mixed.v
        )

    @property
    def h_abl(self) -> Array:
        return jnp.where(
            self.active_bl.is_night,
            self.active_bl.sbl.h_sbl,
            self.active_bl.mixed.h_abl,
        )

    @property
    def wstar(self) -> Array:
        return jnp.where(self.active_bl.is_night, 0.0, self.active_bl.mixed.wstar)

    @property
    def thetav(self) -> Array:
        return jnp.where(
            self.active_bl.is_night,
            self.active_bl.sbl.thetav,
            self.active_bl.mixed.thetav,
        )

    @property
    def top_T(self) -> Array:
        return jnp.where(
            self.active_bl.is_night,
            self.active_bl.sbl.theta - (cst.g / cst.cp) * self.active_bl.sbl.h_sbl,
            self.active_bl.mixed.top_T,
        )

    @property
    def top_p(self) -> Array:
        return jnp.where(
            self.active_bl.is_night,
            self.active_bl.sbl.surf_pressure
            - cst.rho * cst.g * self.active_bl.sbl.h_sbl,
            self.active_bl.mixed.top_p,
        )

    @property
    def wthetav(self) -> Array:
        sbl_wthetav = (
            self.active_bl.sbl.wtheta
            + 0.61 * self.active_bl.sbl.theta * self.active_bl.sbl.wq
        )
        return jnp.where(
            self.active_bl.is_night, sbl_wthetav, self.active_bl.mixed.wthetav
        )

    @property
    def wqe(self) -> Array:
        return jnp.where(self.active_bl.is_night, 0.0, self.active_bl.mixed.wqe)

    @property
    def dq(self) -> Array:
        return jnp.where(self.active_bl.is_night, 0.0, self.active_bl.mixed.dq)

    @property
    def dz_h(self) -> Array:
        return jnp.where(self.active_bl.is_night, 0.0, self.active_bl.mixed.dz_h)

    @property
    def deltaCO2(self) -> Array:
        return jnp.where(self.active_bl.is_night, 0.0, self.active_bl.mixed.deltaCO2)

    @property
    def wCO2e(self) -> Array:
        return jnp.where(self.active_bl.is_night, 0.0, self.active_bl.mixed.wCO2e)

    @property
    def cc_mf(self) -> Array:
        return jnp.where(self.active_bl.is_night, 0.0, self.clouds.cc_mf)

    @property
    def cc_qf(self) -> Array:
        return jnp.where(self.active_bl.is_night, 0.0, self.clouds.cc_qf)

    @property
    def wCO2M(self) -> Array:
        return jnp.where(self.active_bl.is_night, 0.0, self.clouds.wCO2M)

    @property
    def cc_frac(self) -> Array:
        return jnp.where(self.active_bl.is_night, 0.0, self.clouds.cc_frac)


StateAlias = AbstractCoupledState[
    RadT,
    LandT,
    DayAndNightAtmosphereState[SurfT, CloudT],
]


class DayAndNightAtmosphereModel(AbstractAtmosphereModel[DayAndNightAtmosphereState]):
    """Atmosphere model with day/night switching.

    Wraps a surface-layer model, a convective mixed-layer model (day),
    a stable boundary-layer model (night), a residual-layer container,
    and a cloud model.  At sunset the mixed layer is copied to the
    residual layer; at sunrise the mixed layer starts from the SBL height.
    """

    def __init__(
        self,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
        sbl_layer: SBLModel,
        residual_layer: ResidualLayerModel,
        clouds: AbstractCloudModel,
    ):
        self.surface_layer = surface_layer
        self.mixed_layer = mixed_layer
        self.sbl_layer = sbl_layer
        self.residual_layer = residual_layer
        self.clouds = clouds

    def init_state(
        self,
        surface: SurfT,
        mixed: BulkState,
        sbl: SBLState,
        residual: ResidualLayerState,
        clouds: CloudT,
        is_night: bool = False,
    ) -> DayAndNightAtmosphereState[SurfT, CloudT]:
        """Initialize the model state.

        Args:
            surface: Initial surface-layer state.
            mixed: Initial convective mixed-layer state.
            sbl: Initial stable boundary-layer state.
            residual: Initial residual-layer state.
            clouds: Initial cloud state.
            is_night: Whether the model starts at night. Default ``False``.

        Returns:
            The initial atmosphere state.
        """
        active_bl = ActiveBLState(
            is_night=jnp.array(is_night),
            mixed=mixed,
            sbl=sbl,
        )
        return DayAndNightAtmosphereState(
            surface=surface,
            active_bl=active_bl,
            residual=residual,
            clouds=clouds,
        )

    def statistics(
        self,
        state: StateAlias,
        t: Array,
    ) -> DayAndNightAtmosphereState:
        """Update diagnostic statistics for the active boundary layer."""
        is_night = state.atmos.active_bl.is_night

        def day_stats(abl):
            temp_atmos = DayOnlyAtmosphereState(
                surface=state.atmos.surface,
                mixed=abl.mixed,
                clouds=state.atmos.clouds,
            )
            temp_state = state.replace(atmos=temp_atmos)
            ml = self.mixed_layer.statistics(temp_state, t)
            return abl.replace(mixed=ml)

        def night_stats(abl):
            sbl = self.sbl_layer.statistics(state)
            return abl.replace(sbl=sbl)

        active_bl = jax.lax.cond(
            is_night, night_stats, day_stats, state.atmos.active_bl
        )
        return state.atmos.replace(active_bl=active_bl)

    def run(
        self,
        state: StateAlias,
    ) -> DayAndNightAtmosphereState:
        # surface layer is always there
        sl_state = self.surface_layer.run(state)

        # determine whether it's day or night
        in_srad = state.rad.in_srad
        net_rad = state.rad.net_rad
        was_night = state.atmos.active_bl.is_night
        clearly_night = (net_rad <= -20.0) & (in_srad < 10.0)
        clearly_day = (in_srad > 20.0) | (net_rad > 20.0) | (state.land.wtheta > 0.001)
        # in the ambiguous zone, stay in the current regime
        is_night = clearly_night | (was_night & ~clearly_day)
        just_became_night = is_night & ~was_night
        just_became_day = ~is_night & was_night

        # mixed layer
        temp_atmos = DayOnlyAtmosphereState(
            surface=sl_state,
            mixed=state.atmos.active_bl.mixed,
            clouds=state.atmos.clouds,
        )
        temp_state = state.replace(atmos=temp_atmos)
        ml_state = self.mixed_layer.run(temp_state)

        # sunrise
        h_residual = state.atmos.residual.h
        ml_state = ml_state.replace(
            h_abl=jnp.where(
                just_became_day,
                jnp.maximum(state.atmos.active_bl.sbl.h_sbl, 100.0),
                ml_state.h_abl,
            ),
        )

        # stable boundary layer
        sbl_state = self.sbl_layer.run(state, h_residual)

        # sunset
        sbl_state = jax.lax.cond(
            just_became_night,
            lambda s: s.replace(
                theta=state.atmos.active_bl.mixed.theta,
                q=state.atmos.active_bl.mixed.q,
                co2=state.atmos.active_bl.mixed.co2,
                u=state.atmos.active_bl.mixed.u,
                v=state.atmos.active_bl.mixed.v,
            ),
            lambda s: s,
            sbl_state,
        )

        # clouds
        cl_state = self.clouds.run(temp_state)

        # residual
        residual = jax.lax.cond(
            just_became_night,
            lambda _: ResidualLayerState(
                theta=ml_state.theta,
                q=ml_state.q,
                co2=ml_state.co2,
                u=ml_state.u,
                v=ml_state.v,
                h=ml_state.h_abl,
                delta_theta=ml_state.deltatheta,
                delta_q=ml_state.dq,
                delta_co2=ml_state.deltaCO2,
                dz_h=ml_state.dz_h,
            ),
            lambda _: state.atmos.residual,
            None,
        )

        # assemble
        active_bl = ActiveBLState(is_night=is_night, mixed=ml_state, sbl=sbl_state)
        return DayAndNightAtmosphereState(
            surface=sl_state,
            active_bl=active_bl,
            residual=residual,
            clouds=cl_state,
        )

    def warmup(
        self,
        radmodel: AbstractRadiationModel,
        landmodel: AbstractLandModel,
        state: StateAlias,
        t: Array,
        dt: float,
        tstart: float,
    ) -> StateAlias:
        """Warmup the atmos by running it for a few timesteps."""
        state = state.replace(atmos=self.statistics(state, t))
        state = state.replace(rad=radmodel.run(state, t, dt, tstart))
        for _ in range(10):
            sl_state = self.surface_layer.run(state)
            atmostate = replace(state.atmos, surface=sl_state)
            state = state.replace(atmos=atmostate)
        landstate = landmodel.run(state)
        state = state.replace(land=landstate)

        net_rad = state.rad.net_rad
        is_night = ((net_rad <= -20.0) | (state.land.wtheta <= 0.001)).item()

        active_bl = state.atmos.active_bl.replace(is_night=jnp.array(is_night))
        state = state.replace(atmos=state.atmos.replace(active_bl=active_bl))

        if not is_night:
            temp_atmos = DayOnlyAtmosphereState(
                surface=state.atmos.surface,
                mixed=state.atmos.active_bl.mixed,
                clouds=state.atmos.clouds,
            )
            temp_state = state.replace(atmos=temp_atmos)
            if not isinstance(self.clouds, NoCloudModel):
                ml_state = self.mixed_layer.run(temp_state)
                atmostate = replace(
                    state.atmos,
                    active_bl=state.atmos.active_bl.replace(mixed=ml_state),
                )
                state = state.replace(atmos=atmostate)
                cl_state = self.clouds.run(temp_state)
                atmostate = replace(state.atmos, clouds=cl_state)
                state = state.replace(atmos=atmostate)
            ml_state = self.mixed_layer.run(temp_state)
            atmostate = replace(
                state.atmos,
                active_bl=state.atmos.active_bl.replace(mixed=ml_state),
            )
            state = state.replace(atmos=atmostate)
        else:
            h_residual = state.atmos.residual.h
            sbl_state = self.sbl_layer.run(state, h_residual)
            sbl_state = self.sbl_layer.integrate(sbl_state, dt)
            atmostate = replace(
                state.atmos,
                active_bl=state.atmos.active_bl.replace(sbl=sbl_state),
            )
            state = state.replace(atmos=atmostate)

        return state

    def integrate(
        self,
        state: DayAndNightAtmosphereState,
        dt: float,
    ) -> DayAndNightAtmosphereState:
        """Integrate the active boundary layer forward in time."""
        is_night = state.active_bl.is_night

        def _integrate_day(abl):
            ml = self.mixed_layer.integrate(abl.mixed, dt)
            return abl.replace(mixed=ml)

        def _integrate_night(abl):
            sbl = self.sbl_layer.integrate(abl.sbl, dt)
            return abl.replace(sbl=sbl)

        active_bl = jax.lax.cond(
            is_night, _integrate_night, _integrate_day, state.active_bl
        )
        return replace(state, active_bl=active_bl)
