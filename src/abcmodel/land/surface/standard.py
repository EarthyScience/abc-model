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
        """Compute the derivative of saturation vapor pressure with respect to temperature ``dqsatdT``.

        Notes:
            Using :func:`~abcmodel.utils.compute_esat`, the derivative of the saturated vapor pressure
            :math:`e_\\text{sat}` with respect to temperature :math:`T` is given by

            .. math::
                \\frac{\\text{d}e_\\text{sat}}{\\text{d} T} =
                e_\\text{sat}\\frac{17.2694(T-237.16)}{(T-35.86)^2},

            which combined with :func:`~abcmodel.utils.compute_qsat` can be used to get

            .. math::
                \\frac{\\text{d}q_{\\text{sat}}}{\\text{d} T} \\approx \\epsilon \\frac{\\frac{\\text{d}e_\\text{sat}}{\\text{d} T}}{p}.
        """
        num = 17.2694 * (theta - 273.16)
        den = (theta - 35.86) ** 2.0
        mult = num / den
        desatdT = esat * mult
        return 0.622 * desatdT / surf_pressure

    def compute_e(self, q: Array, surf_pressure: Array) -> Array:
        """Compute the vapor pressure ``e``.

        Notes:
            This function uses the same formula used in :func:`~abcmodel.utils.compute_esat`,
            but now factoring the vapor pressure :math:`e` as a function of specific humidity :math:`q`
            and surface pressure :math:`p`, which give us

            .. math::
                e = q \\cdot p / 0.622.
        """
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
        """Compute the skin temperature ``surf_temp``.

        Notes:
            The skin temperature is obtained by solving the surface energy balance

            .. math::
                R_n = H + LE_{\\text{veg}} + LE_{\\text{liq}} + LE_{\\text{soil}} + G

            where :math:`R_n` is the net rad,
            :math:`H` is the sensible heat flux (see :meth:`~.StandardSurfaceModel.compute_hf`),
            :math:`LE_{\\text{veg}}` is the latent heat flux from vegetation (see :meth:`~.StandardSurfaceModel.compute_le_veg`),
            :math:`LE_{\\text{liq}}` is the latent heat flux from dew on leaves (see :meth:`~.StandardSurfaceModel.compute_le_liq`),
            :math:`LE_{\\text{soil}}` is the latent heat flux from the soil (see :meth:`~.StandardSurfaceModel.compute_le_soil`)
            and :math:`G` is the ground heat flux (see :meth:`~.StandardSurfaceModel.compute_gf`).

            The equation is solved for the skin temperature :math:`T_s`
            by factoring out :math:`T_s` from the above, giving us

            .. math::
                T_s = \\frac{
                    R_n + \\frac{\\rho c_p}{r_a} \\theta
                    + c_{\\text{veg}} (1-c_{\\text{liq}}) \\frac{\\rho L_v}{r_a + r_s} (\\frac{\\text{d}q_{\\text{sat}}}{\\text{d}T} \\theta - q_{\\text{sat}} + q)
                    + (1-c_{\\text{veg}}) \\frac{\\rho L_v}{r_a + r_{s,\\text{soil}}} (\\frac{\\text{d}q_{\\text{sat}}}{\\text{d}T} \\theta - q_{\\text{sat}} + q)
                    + c_{\\text{veg}} c_{\\text{liq}} \\frac{\\rho L_v}{r_a} (\\frac{\\text{d}q_{\\text{sat}}}{\\text{d}T} \\theta - q_{\\text{sat}} + q)
                    + \\Lambda T_{\\text{soil}}
                }{
                    \\frac{\\rho c_p}{r_a}
                    + c_{\\text{veg}} (1-c_{\\text{liq}}) \\frac{\\rho L_v}{r_a + r_s} \\frac{\\text{d}q_{\\text{sat}}}{\\text{d}T}
                    + (1-c_{\\text{veg}}) \\frac{\\rho L_v}{r_a + r_{s,\\text{soil}}} \\frac{\\text{d}q_{\\text{sat}}}{\\text{d}T}
                    + c_{\\text{veg}} c_{\\text{liq}} \\frac{\\rho L_v}{r_a} \\frac{\\text{d}q_{\\text{sat}}}{\\text{d}T}
                    + \\Lambda
                }.

            The terms computed in the equation above in each related function energy flux related method.
            This approach ensures that the computed skin temperature is consistent with the partitioning of
            energy fluxes as calculated by the other methods in this class.
        """
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
        """Compute the latent heat flux (transpiration) from vegetation ``le_veg``.

        Notes:
            The latent heat flux is given by

            .. math::
                LE_{\\text{veg}} = \\frac{\\rho L_v}{r_a+r_s}(q_{\\text{sat}}(T_s)-⟨q⟩),

            where :math:`\\rho` is the density of air, :math:`L_v` is the latent heat of vaporization,
            :math:`r_a` is the aerodynamic resistance, :math:`r_s` is the soil resistance,
            :math:`q_{\\text{sat}}(T_s)` is the saturation specific humidity at surface temperature,
            :math:`⟨q⟩` is the specific humidity at the surface.

            :math:`q_{\\text{sat}}(T_s)` has very short time-scales because of the small heat capacity
            (excluding vegetation) of the surface layer and is hard to measure. Consequently,
            we get :math:`q_{\\text{sat}}(T_s)` implicitly using

            .. math::
                q_{\\text{sat}}(T_s) = \\frac{\\text{d}q_{\\text{sat}}}{\\text{d}T}(\\theta_s-\\theta),

            where :math:`\\theta_s` and :math:`\\theta` are the potential temperature of the surface layer and mixed layer, respectively.

            In the end, we scale the latent heat flux by the vegetation cover fraction :math:`c_{\\text{veg}}`
            and the liquid water content :math:`c_{\\text{liq}}` and return

            .. math::
                c_{\\text{veg}}(1-c_{\\text{liq}})LE_{\\text{veg}}.

        References:
            Equation 9.15 from the CLASS book.
        """
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
        """Compute the latent heat flux on the leaf (dew) ``le_liq``.

        Notes:
            We proceed just like in :meth:`~.StandardSurfaceModel.compute_le_veg`, but omitting vegetation's resistance :math:`r_s`,
            with the assumption that water at the leaf is ready to be evaporated, giving us

        .. math::
            LE_{\\text{liq}} = \\frac{\\rho L_v}{r_a}(q_{\\text{sat}}(T_s)-⟨q⟩).

        In the end, we scale the result by the fraction of liquid water content :math:`c_{\\text{liq}}`
        and the fraction of vegetation :math:`c_{\\text{veg}}`.

        References:
            Equation 9.18 from the CLASS book.
        """
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
        """Compute the latent heat flux on the soil (evaporation) ``le_soil``.

        Notes:
            We proceed just like in :meth:`~.StandardSurfaceModel.compute_le_veg`, but instead of considering resistance from
            the vegetation, we consider the resistance from the soil :math:`r_{soil}`, giving us

        .. math::
            LE_{\\text{soil}} = \\frac{\\rho L_v}{r_a + r_{soil}}(q_{\\text{sat}}(T_s)-⟨q⟩)

        In the end, we scale the result by the fraction of soil :math:`c_{\\text{soil}} = 1 - c_{\\text{veg}}`.

        References:
            Equation 9.21 from the CLASS book.
        """
        term = dqsatdT * (surf_temp - theta) + qsat - q
        le_soil = cst.rho * cst.lv / (ra + rssoil) * term
        frac = 1.0 - cveg
        return frac * le_soil

    def compute_le(self, le_soil: Array, le_veg: Array, le_liq: Array) -> Array:
        """Compute the evapotranspiration (latent heat flux) ``le``.

        Notes:
            The latent heat flux is the sum of transpiration from
            vegetation, bare soil evaporation, and wet-leaf evaporation,
            clipped to non-negative values

            .. math::
                \\text{LE} = \\max\bigl(\\text{LE}_{\\text{soil}} +
                    \\text{LE}_{\\text{veg}} + \\text{LE}_{\\text{liq}},; 0\\bigr)
        """
        return jnp.clip(le_soil + le_veg + le_liq, 0.0, None)

    def compute_hf(self, surf_temp: Array, theta: Array, ra: Array) -> Array:
        """Compute the sensible heat flux ``hf``.

        Notes:
            The sensible heat flux is given by

            .. math::

                H = \\frac{\\rho c_p}{r_a} (T_s - \\theta),

            where :math:`\\rho` is the air density, :math:`c_p` is the specific heat capacity of air,
            :math:`r_a` is the aerodynamic resistance, :math:`T_s` is the surface temperature and
            :math:`\\theta` is the mixed layer air potential temperature.

        References:
            Equation 9.13 from the CLASS book, but why are we using :math:`T_s` instead of :math:`\\theta_s`?
            Probably because the variations of pressure are not significant enough.
        """
        return cst.rho * cst.cp / ra * (surf_temp - theta)

    def compute_gf(self, surf_temp: Array, temp_soil: Array) -> Array:
        """Compute the ground heat flux ``gf``.

        Notes:
            The ground heat flux is given by

            .. math::

                G = \\Lambda (T_s - T_{soil}),

            where :math:`\\Lambda` is the conductivity of the skin layer,
            :math:`T_s` is the surface temperature and
            :math:`T_{soil}` is the soil temperature.

        References:
            Equation 9.33 from the CLASS book.
        """
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
        """Compute the potential latent heat flux ``le_pot``.

        Notes:
            The potential latent heat flux is given by

            .. math::

                LE_{\\text{pot}} = \\frac{
                \\frac{\\text{d}q_{sat}}{\\text{d} T} (R_n - G)
                + \\frac{\\rho c_p}{r_a} (q_{\\text{sat}} - q)
                }{
                \\frac{\\text{d}q_{sat}}{\\text{d} T} + \\frac{\\rho c_p}{L_v}
                },

            which is the Penman-Monteith equation assuming no soil resistance.

        References:
            Equation 9.16 from the CLASS book.
        """
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
        """Compute the reference latent heat flux ``le_ref``.

        Notes:
            The reference latent heat flux is given by

            .. math::

                LE_{\\text{ref}} = \\frac{
                \\frac{\\text{d}q_{sat}}{\\text{d} T} (R_n - G)
                + \\frac{\\rho c_p}{r_a} (q_{\\text{sat}} - q)
                }{
                \\frac{\\text{d}q_{sat}}{\\text{d} T} + \\frac{\\rho c_p}{L_v}(
                1 + \\frac{r_{s,\\text{min}}}{\\text{LAI} \\cdot r_a}
                )
                },

            which is the Penman-Monteith equation assuming that the soil resistance is given by
            :math:`r_{s,\\text{min}} / \\text{LAI}`, i.e., no correction functions are applied.

        References:
            Equation 9.16 from the CLASS book.
        """
        rad_term = dqsatdT * (net_rad - gf)
        aerodynamic_term = cst.rho * cst.cp / ra * (qsat - q)
        den1 = dqsatdT
        den2 = cst.cp / cst.lv * (1.0 + self.rsmin / self.lai / ra)
        return (rad_term + aerodynamic_term) / (den1 + den2)

    def compute_wtheta(self, hf: Array) -> Array:
        """Compute the kinematic heat flux ``wtheta``.

        Notes:
            The kinematic heat flux :math:`\\overline{(w'\\theta')}_s` is directly related to the
            sensible heat flux :math:`H` through

            .. math::
                \\overline{(w'\\theta')}_s = \\frac{H}{\\rho c_p},

            where :math:`\\rho` is the density of air and
            :math:`c_p` is the specific heat capacity of air at constant pressure.
        """
        return hf / (cst.rho * cst.cp)

    def compute_wq(self, le: Array) -> Array:
        """Compute the kinematic moisture flux ``wq``.

        Notes:
            The kinematic moisture flux :math:`\\overline{(w'q')}_s` is directly related to the
            latent heat flux :math:`LE` through

            .. math::
                \\overline{(w'q')}_s = \\frac{LE}{\\rho L_v},

            where :math:`\\rho` is the density of air and
            :math:`L_v` is the latent heat of vaporization.
        """
        return le / (cst.rho * cst.lv)

    def compute_vpd(self, q: Array, qsat: Array) -> Array:
        """Compute the vapour pressure deficit ``vpd``.

        Notes:
            The vapour pressure deficit is the difference between the
            saturation specific humidity and the actual specific humidity

            .. math::
                D_q = q_{\text{sat}} - q

            where :math:`q_{\text{sat}}` is the saturation specific
            humidity and :math:`q` is the actual specific humidity.
        """
        return qsat - q
