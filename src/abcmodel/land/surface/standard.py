from dataclasses import dataclass, field, replace

import jax.numpy as jnp
from jax import Array

from ...abstracts import AbstractCoupledState
from ...utils import PhysicalConstants as cst
from ...utils import compute_esat, compute_qsat
from ..abstracts import AbstractSurfaceModel, AbstractSurfaceState


@dataclass
class StandardSurfaceState(AbstractSurfaceState):
    """Standard surface state."""

    alpha: Array = field(
        metadata={
            "label": r"$\alpha$",
            "unit": "-",
            "description": "Surface albedo",
        }
    )
    """Surface albedo [-]."""
    surf_temp: Array = field(
        metadata={
            "label": r"$T_{surf}$",
            "unit": "K",
            "description": "Surface temperature",
        }
    )
    """Surface temperature [K]."""
    esat: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$e_{sat}$",
            "unit": "Pa",
            "description": "Saturation vapor pressure",
        },
    )
    """Saturation vapor pressure [Pa]."""
    qsat: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$q_{sat}$",
            "unit": "kg kg^{-1}",
            "description": "Saturation specific humidity",
        },
    )
    """Saturation specific humidity [kg/kg]."""
    dqsatdT: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$dq_{sat}/dT$",
            "unit": "kg kg^{-1} K^{-1}",
            "description": "Derivative of saturation specific humidity",
        },
    )
    """Derivative of saturation specific humidity with respect to temperature [kg/kg/K]."""
    e: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$e$",
            "unit": "Pa",
            "description": "Vapor pressure",
        },
    )
    """Vapor pressure [Pa]."""
    qsatsurf: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$q_{sat}(T_s)$",
            "unit": "kg kg^{-1}",
            "description": "Saturation specific humidity at surface",
        },
    )
    """Saturation specific humidity at surface temperature [kg/kg]."""
    le_veg: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$LE_{veg}$",
            "unit": "W m^{-2}",
            "description": "Latent heat flux from vegetation",
        },
    )
    """Latent heat flux from vegetation [W m-2]."""
    le_liq: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$LE_{liq}$",
            "unit": "W m^{-2}",
            "description": "Latent heat flux from liquid water",
        },
    )
    """Latent heat flux from liquid water [W m-2]."""
    le_soil: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$LE_{soil}$",
            "unit": "W m^{-2}",
            "description": "Latent heat flux from soil",
        },
    )
    """Latent heat flux from soil [W m-2]."""
    le: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$LE$",
            "unit": "W m^{-2}",
            "description": "Total latent heat flux",
        },
    )
    """Total latent heat flux [W m-2]."""
    hf: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$H$",
            "unit": "W m^{-2}",
            "description": "Sensible heat flux",
        },
    )
    """Sensible heat flux [W m-2]."""
    gf: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$G$",
            "unit": "W m^{-2}",
            "description": "Ground heat flux",
        },
    )
    """Ground heat flux [W m-2]."""
    le_pot: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$LE_{pot}$",
            "unit": "W m^{-2}",
            "description": "Potential latent heat flux",
        },
    )
    """Potential latent heat flux [W m-2]."""
    le_ref: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$LE_{ref}$",
            "unit": "W m^{-2}",
            "description": "Reference latent heat flux",
        },
    )
    """Reference latent heat flux [W m-2]."""
    vpd: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$VPD$",
            "unit": "Pa",
            "description": "Vapor pressure deficit",
        },
    )
    """Vapor pressure deficit [Pa]."""
    wtheta: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$(w'\theta')_s$",
            "unit": "K m s^{-1}",
            "description": "Kinematic heat flux",
        },
    )
    """Kinematic heat flux [K m/s]."""
    wq: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$(w'q')_s$",
            "unit": "kg kg^{-1} m s^{-1}",
            "description": "Kinematic moisture flux",
        },
    )
    """Kinematic moisture flux [kg/kg m/s]."""


class StandardSurfaceModel(AbstractSurfaceModel[StandardSurfaceState]):
    """Standard surface model calculating skin temperature and energy fluxes.

    Args:
        lam: thermal diffusivity of the soil/skin layer [W m-1 K-1]. Default is 5.9.
        rsmin: minimum stomatal resistance [s m-1]. Default is 110.0.
        lai: leaf area index [m2 m-2]. Default is 2.0.
    """

    def __init__(
        self,
        lam: float = 5.9,
        rsmin: float = 110.0,
        lai: float = 2.0,
    ):
        self.lam = lam
        self.rsmin = rsmin
        self.lai = lai

    def init_state(
        self,
        alpha: float = 0.25,
        surf_temp: float = 290.0,
    ) -> StandardSurfaceState:
        """Initialize the surface state.

        Args:
            alpha: surface albedo [-]. Default is 0.25.
            surf_temp: surface skin temperature [K]. Default is 290.0.

        Returns:
            The initialized StandardSurfaceState.
        """
        return StandardSurfaceState(
            alpha=jnp.array(alpha),
            surf_temp=jnp.array(surf_temp),
        )

    def run(self, state: AbstractCoupledState) -> StandardSurfaceState:
        """Compute surface energy balance, skin temperature, and fluxes."""
        land = state.land
        atmos = state.atmos
        ra = atmos.ra

        esat = compute_esat(atmos.theta)
        qsat = compute_qsat(atmos.theta, atmos.surf_pressure)
        dqsatdT = self.compute_dqsatdT(esat, atmos.theta, atmos.surf_pressure)
        e = self.compute_e(atmos.q, atmos.surf_pressure)

        rs = land.biosphere.rs
        cliq = land.biosphere.cliq
        cveg = land.biosphere.cveg

        rssoil = land.soil.rssoil
        temp_soil = land.soil.temp_soil

        surf_temp = self.compute_skin_temperature(
            state.net_rad,
            atmos.theta,
            atmos.q,
            qsat,
            dqsatdT,
            ra,
            rs,
            rssoil,
            cliq,
            temp_soil,
            cveg,
        )
        qsatsurf = compute_qsat(surf_temp, atmos.surf_pressure)

        le_veg = self.compute_le_veg(
            surf_temp,
            atmos.theta,
            atmos.q,
            qsat,
            dqsatdT,
            ra,
            rs,
            cliq,
            cveg,
        )
        le_liq = self.compute_le_liq(
            surf_temp,
            atmos.theta,
            atmos.q,
            qsat,
            dqsatdT,
            ra,
            cliq,
            cveg,
        )
        le_soil = self.compute_le_soil(
            surf_temp,
            atmos.theta,
            atmos.q,
            qsat,
            dqsatdT,
            ra,
            rssoil,
            cveg,
        )
        le = self.compute_le(le_soil, le_veg, le_liq)
        hf = self.compute_hf(surf_temp, atmos.theta, ra)
        gf = self.compute_gf(surf_temp, temp_soil)

        le_pot = self.compute_le_pot(
            state.net_rad,
            gf,
            dqsatdT,
            qsat,
            atmos.q,
            ra,
        )
        le_ref = self.compute_le_ref(
            state.net_rad,
            gf,
            dqsatdT,
            qsat,
            atmos.q,
            ra,
        )

        wtheta = self.compute_wtheta(hf)
        wq = self.compute_wq(le)
        vpd = self.compute_vpd(atmos.q, qsat)

        return replace(
            land.surface,
            esat=esat,
            qsat=qsat,
            dqsatdT=dqsatdT,
            e=e,
            surf_temp=surf_temp,
            qsatsurf=qsatsurf,
            le_veg=le_veg,
            le_liq=le_liq,
            le_soil=le_soil,
            le=le,
            hf=hf,
            gf=gf,
            le_pot=le_pot,
            le_ref=le_ref,
            vpd=vpd,
            wtheta=wtheta,
            wq=wq,
        )

    def compute_dqsatdT(self, esat: Array, theta: float, surf_pressure: float) -> Array:
        """Compute derivative of saturation vapor pressure with respect to temperature."""
        num = 17.2694 * (theta - 273.16)
        den = (theta - 35.86) ** 2.0
        mult = num / den
        desatdT = esat * mult
        return 0.622 * desatdT / surf_pressure

    def compute_e(self, q: Array, surf_pressure: Array) -> Array:
        """Compute the vapor pressure."""
        return q * surf_pressure / 0.622

    def compute_skin_temperature(
        self,
        net_rad: Array,
        theta: Array,
        q: Array,
        qsat: Array,
        dqsatdT: Array,
        ra: Array,
        rs: Array,
        rssoil: Array,
        cliq: Array,
        temp_soil: Array,
        cveg: Array,
    ) -> Array:
        """Compute skin temperature surf_temp."""
        return (
            net_rad
            + cst.rho * cst.cp / ra * theta
            + cveg
            * (1.0 - cliq)
            * cst.rho
            * cst.lv
            / (ra + rs)
            * (dqsatdT * theta - qsat + q)
            + (1.0 - cveg)
            * cst.rho
            * cst.lv
            / (ra + rssoil)
            * (dqsatdT * theta - qsat + q)
            + cveg * cliq * cst.rho * cst.lv / ra * (dqsatdT * theta - qsat + q)
            + self.lam * temp_soil
        ) / (
            cst.rho * cst.cp / ra
            + cveg * (1.0 - cliq) * cst.rho * cst.lv / (ra + rs) * dqsatdT
            + (1.0 - cveg) * cst.rho * cst.lv / (ra + rssoil) * dqsatdT
            + cveg * cliq * cst.rho * cst.lv / ra * dqsatdT
            + self.lam
        )

    def compute_le_veg(
        self,
        surf_temp: Array,
        theta: Array,
        q: Array,
        qsat: Array,
        dqsatdT: Array,
        ra: Array,
        rs: Array,
        cliq: Array,
        cveg: Array,
    ) -> Array:
        """Compute latent heat flux (transpiration) from vegetation."""
        term = dqsatdT * (surf_temp - theta) + qsat - q
        le_veg = cst.rho * cst.lv / (ra + rs) * term
        frac = (1.0 - cliq) * cveg
        return frac * le_veg

    def compute_le_liq(
        self,
        surf_temp: Array,
        theta: Array,
        q: Array,
        qsat: Array,
        dqsatdT: Array,
        ra: Array,
        cliq: Array,
        cveg: Array,
    ) -> Array:
        """Compute latent heat flux on the leaf (dew) le_liq."""
        term = dqsatdT * (surf_temp - theta) + qsat - q
        le_liq = cst.rho * cst.lv / ra * term
        frac = cliq * cveg
        return frac * le_liq

    def compute_le_soil(
        self,
        surf_temp: Array,
        theta: Array,
        q: Array,
        qsat: Array,
        dqsatdT: Array,
        ra: Array,
        rssoil: Array,
        cveg: Array,
    ) -> Array:
        """Compute latent heat flux on the soil (evaporation) le_soil."""
        term = dqsatdT * (surf_temp - theta) + qsat - q
        le_soil = cst.rho * cst.lv / (ra + rssoil) * term
        frac = 1.0 - cveg
        return frac * le_soil

    def compute_le(self, le_soil: Array, le_veg: Array, le_liq: Array) -> Array:
        """Compute the total evapotranspiration (latent heat flux) le."""
        return jnp.clip(le_soil + le_veg + le_liq, 0.0, None)

    def compute_hf(self, surf_temp: Array, theta: Array, ra: Array) -> Array:
        """Compute sensible heat flux hf."""
        return cst.rho * cst.cp / ra * (surf_temp - theta)

    def compute_gf(self, surf_temp: Array, temp_soil: Array) -> Array:
        """Compute ground heat flux gf."""
        return self.lam * (surf_temp - temp_soil)

    def compute_le_pot(
        self,
        net_rad: Array,
        gf: Array,
        dqsatdT: Array,
        qsat: Array,
        q: Array,
        ra: Array,
    ) -> Array:
        """Compute potential latent heat flux."""
        rad_term = dqsatdT * (net_rad - gf)
        aerodynamic_term = cst.rho * cst.cp / ra * (qsat - q)
        denominator = dqsatdT + cst.cp / cst.lv
        return (rad_term + aerodynamic_term) / denominator

    def compute_le_ref(
        self,
        net_rad: Array,
        gf: Array,
        dqsatdT: Array,
        qsat: Array,
        q: Array,
        ra: Array,
    ) -> Array:
        """Compute reference latent heat flux."""
        rad_term = dqsatdT * (net_rad - gf)
        aerodynamic_term = cst.rho * cst.cp / ra * (qsat - q)
        den1 = dqsatdT
        den2 = cst.cp / cst.lv * (1.0 + self.rsmin / self.lai / ra)
        return (rad_term + aerodynamic_term) / (den1 + den2)

    def compute_wtheta(self, hf: Array) -> Array:
        """Compute kinematic heat flux."""
        return hf / (cst.rho * cst.cp)

    def compute_wq(self, le: Array) -> Array:
        """Compute kinematic moisture flux."""
        return le / (cst.rho * cst.lv)

    def compute_vpd(self, q: Array, qsat: Array) -> Array:
        """Compute vapor pressure deficit."""
        return qsat - q
