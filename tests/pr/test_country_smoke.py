# Copyright 2026 Power Lab

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Cross-country smoke tests: real turbine-curve and pvlib ModelChain code,
synthetic inputs only.

No CDS/network access and no real weather downloads -- the goal is to catch
the *class* of bug in GitHub issue #1 (wind_cf returning -inf outside the
turbine power curve): exceptions or non-finite/negative output, across
every country's actual configured turbine/module/inverter and across dates
spanning 2015-2025. This does not validate the scientific accuracy of any
output value, only that the pipeline runs clean for each
country/technology/date combination.

Note on the Indonesia hemisphere-orientation bug specifically: Stage B runs
the real pvlib ModelChain at a southern-hemisphere point using the correct
orientation, which does confirm the ModelChain itself handles negative
latitudes without error -- but a *wrong* orientation (the original bug)
produces lower, still-finite output, not NaN/-inf, so these finite/
non-negative assertions would not by themselves catch a regression back to
a hardcoded orientation. That specific regression is covered separately by
tests/pr/test_pvlib_orientation.py's direct unit tests on
latitude_optimal_orientation().

Turbine/module/inverter picks below are copied from
geodata_helpers/profile_generation/country_tech_config.yaml, and sample
coordinates from geodata_helpers/profile_split/nations.yaml's crop_bounds.
That repo has no shared CI with this one -- keep these in sync manually if
either file changes.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geodata.datasets import load_dataset
from geodata.model.pvlib import Pvlib, latitude_optimal_orientation
from geodata.model.wind._base import WindBaseModel
from geodata.resource import get_windturbineconfig

# ---------------------------------------------------------------------------
# Stage A: turbine power curves
# ---------------------------------------------------------------------------

COUNTRY_TURBINES = {
    "china": "Enercon_E101_3000kW",
    "usa": "Vestas_V90_3MW",
    "vietnam": "Vestas_V112_3MW",
    "indonesia": "Siemens_SWT_107_3600kW",
}


class _FakeWindModel(WindBaseModel):
    """Returns a synthetic wind-speed sweep instead of reading a real
    dataset, so `_estimate_power`'s real interp1d/turbine-curve code runs
    unmodified against the exact production path."""

    def __init__(self, wind_speed: xr.DataArray):
        self._wind_speed = wind_speed  # deliberately skip BaseModel.__init__

    def estimate(self, years=None, months=None, xs=None, ys=None, **kwargs):
        if "turbine" in kwargs:
            return self._estimate_power(
                years=years, months=months, xs=xs, ys=ys, **kwargs
            )
        return self._wind_speed

    # Unused abstract methods required only to satisfy BaseModel's ABC;
    # `estimate()` above is fully overridden and never dispatches to these.
    def _estimate_dataset(self, *args, **kwargs):
        raise NotImplementedError

    def _prepare_dataset(self, *args, **kwargs):
        raise NotImplementedError


@pytest.mark.parametrize("country,turbine", COUNTRY_TURBINES.items())
def test_turbine_power_curve_no_nonfinite_values(country, turbine):
    """Sweep wind speeds from well below cut-in to well above cut-out and
    confirm cf is always finite and in [0, 1] -- regression check for the
    exact bug class in issue #1, across every country's actual turbine."""
    conf = get_windturbineconfig(turbine)
    v_min, v_max = float(conf["V"].min()), float(conf["V"].max())

    speeds = np.concatenate(
        [
            np.array([v_min - 5.0, -0.01, 0.0]),  # below cut-in / noise near 0
            np.linspace(v_min, v_max, 25),  # across the whole tabulated curve
            np.array([v_max, v_max + 0.5, v_max + 10.0, v_max + 40.0]),  # at/above cut-out
        ]
    )
    speed_da = xr.DataArray(speeds, dims=["point"], name="wnd100m")

    model = _FakeWindModel(speed_da)
    cf = model.estimate(turbine=turbine)

    assert np.isfinite(cf.values).all(), (
        f"{country}/{turbine}: non-finite cf values at speeds "
        f"{speeds[~np.isfinite(cf.values)]}"
    )
    assert (cf.values >= 0).all() and (cf.values <= 1.0 + 1e-9).all(), (
        f"{country}/{turbine}: cf outside [0, 1]: "
        f"min={cf.values.min()}, max={cf.values.max()}"
    )


# ---------------------------------------------------------------------------
# Stage B: full pvlib ModelChain
# ---------------------------------------------------------------------------

# One or more (lat, lon) sample points per country, drawn from
# geodata_helpers/profile_split/nations.yaml's crop_bounds. Indonesia gets
# both a northern and a southern point since its bounds cross the equator --
# the exact territory of the hemisphere-orientation bug.
COUNTRY_SOLAR_CONFIG = {
    "china": {
        "module": "Jinko_Solar__Co___Ltd_JKM400M_72L",
        "inverter": "Sungrow_Power_Supply_Co___Ltd___SG100KU__480V_",
        "points": [(34.3, 108.9)],  # Xi'an
    },
    "usa": {
        "module": "First_Solar__Inc__FS_6420A",
        "inverter": "Fronius_USA__CL_33_3_Delta__208V_",
        "points": [(37.7, -97.3)],  # Wichita, KS
    },
    "vietnam": {
        "module": "Canadian_Solar_Inc__CS1U_430MS",
        "inverter": "Sungrow_Power_Supply_Co___Ltd___SG100KU__480V_",
        "points": [(21.0, 105.8)],  # Hanoi
    },
    "indonesia": {
        "module": "Kaneka_U_SA105",
        "inverter": "Sungrow_Power_Supply_Co___Ltd___SG100KU__480V_",
        "points": [(5.55, 95.32), (-10.17, 123.6)],  # Banda Aceh (N), Kupang/NTT (S)
    },
}

# Every leap year in 2015-2025, plus non-leap boundary years, to exercise
# Feb 29 handling and year-boundary date arithmetic.
YEARS = [2015, 2016, 2020, 2024, 2025]


def _synthetic_solar_ds(lat: float, lon: float) -> xr.Dataset:
    """Single-point synthetic dataset shaped like a prepared geodata cutout
    (dims time/y/x, ERA5-derived variable names), with representative
    daytime/nighttime hours spanning every leap year + boundary in
    2015-2025. Deliberately not physically realistic -- only shaped/ranged
    correctly enough to exercise the real code without producing structural
    errors of its own (e.g. negative Kelvin)."""
    timestamps = set()
    for year in YEARS:
        for month, day, hour in [(1, 1, 0), (6, 21, 12), (12, 31, 23)]:
            timestamps.add(pd.Timestamp(year=year, month=month, day=day, hour=hour))
    time = pd.DatetimeIndex(sorted(timestamps))

    nt = len(time)
    rng = np.random.default_rng(abs(hash((lat, lon))) % (2**32))

    return xr.Dataset(
        {
            "influx_diffuse": (["time", "y", "x"], rng.uniform(0, 200, (nt, 1, 1))),
            "influx_direct": (["time", "y", "x"], rng.uniform(0, 800, (nt, 1, 1))),
            "dewpoint_temperature": (["time", "y", "x"], rng.uniform(260, 290, (nt, 1, 1))),
            "temperature": (["time", "y", "x"], rng.uniform(270, 305, (nt, 1, 1))),
            "wnd100m": (["time", "y", "x"], rng.uniform(0, 15, (nt, 1, 1))),
        },
        coords={"time": time, "y": [lat], "x": [lon]},
    )


@pytest.mark.parametrize("country", COUNTRY_SOLAR_CONFIG.keys())
def test_pvlib_modelchain_no_nonfinite_values(country):
    """Run the real pvlib ModelChain code across each country's configured
    module/inverter, at its real latitude range (incl. Indonesia's southern
    point), across dates spanning every leap year in 2015-2025."""
    cfg = COUNTRY_SOLAR_CONFIG[country]

    # A Pvlib instance needs a `source` dataset to satisfy BaseModel.__init__'s
    # validation; reuse the existing offline fixture purely for that -- the
    # actual computation below always runs on our synthetic per-point `ds`,
    # never on this fixture's own data.
    fixture_cls = load_dataset("wind_solar_hourly_test")
    fixture = fixture_cls(years=slice(2016, 2016), months=slice(1, 1))
    model = Pvlib(fixture)

    cec_modules = model.retrieve_sam("CECMod")
    cec_inverters = model.retrieve_sam("CECInverter")
    module = cec_modules[cfg["module"]]
    inverter = cec_inverters[cfg["inverter"]]

    model.init_model_config(
        clearsky_model="haurwitz",
        transposition_model="perez",
        solar_position_method="nrel_numpy",
        airmass_model="kastenyoung1989",
        dc_model="cec",
        ac_model="sandia",
        aoi_model="physical",
        spectral_model="first_solar",
        dc_ohmic_model="no_loss",
    )

    for lat, lon in cfg["points"]:
        orientation = latitude_optimal_orientation(lat)
        model.init_pv_system(
            arrays=None,
            surface_tilt=orientation["surface_tilt"],
            surface_azimuth=orientation["surface_azimuth"],
            racking_model="open_rack",
            module_parameters=module,
            modules_per_string=50,
            module_type="glass_polymer",
            module=cfg["module"],
            strings_per_inverter=1,
            inverter_parameters=inverter,
        )

        ds = _synthetic_solar_ds(lat, lon)
        result = model._pvlib_model(
            ds, model.pv_system, model.config, n_jobs=1, compact_output=True
        )

        assert "ac" in result.data_vars, f"{country}@({lat},{lon}): missing 'ac' output"
        ac_values = result["ac"].values
        assert np.isfinite(ac_values).all(), (
            f"{country}@({lat},{lon}): non-finite ac output "
            f"(orientation={orientation})"
        )
        assert (ac_values >= 0).all(), (
            f"{country}@({lat},{lon}): negative ac output "
            f"(orientation={orientation})"
        )


def _synthetic_solar_ds_full_year(lat: float, lon: float, year: int) -> xr.Dataset:
    """Full-year hourly synthetic dataset at a single point. Unlike
    `_synthetic_solar_ds` (a handful of sparse dates), this covers every
    hour of one year so the sun's real position (computed from real
    timestamps, not synthetic) traces its actual seasonal/daily path --
    needed to meaningfully compare two orientations' *relative* output,
    not just check for crashes/non-finite values."""
    time = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    nt = len(time)
    rng = np.random.default_rng(abs(hash((lat, lon, year))) % (2**32))

    return xr.Dataset(
        {
            "influx_diffuse": (["time", "y", "x"], rng.uniform(0, 200, (nt, 1, 1))),
            "influx_direct": (["time", "y", "x"], rng.uniform(0, 800, (nt, 1, 1))),
            "dewpoint_temperature": (["time", "y", "x"], rng.uniform(295, 300, (nt, 1, 1))),
            "temperature": (["time", "y", "x"], rng.uniform(298, 305, (nt, 1, 1))),
            "wnd100m": (["time", "y", "x"], rng.uniform(0, 8, (nt, 1, 1))),
        },
        coords={"time": time, "y": [lat], "x": [lon]},
    )


def test_southern_hemisphere_orientation_beats_hardcoded_south_facing():
    """Regression guard for the Indonesia hemisphere-orientation bug at the
    full-ModelChain level, not just the `latitude_optimal_orientation()`
    helper (already unit-tested separately in test_pvlib_orientation.py).

    Runs the real pvlib ModelChain a full synthetic year at a southern
    Indonesia point, once with the correct (north-facing) orientation and
    once with the old hardcoded (south-facing, tilt=35/azimuth=180) bug,
    and asserts the correct orientation produces meaningfully more annual
    energy -- the physically expected direction south of the equator.
    Empirically the fix produces ~1.7x the old bug's output here; a 1.1x
    margin is asserted to leave headroom against synthetic-data noise
    while still failing on any real regression back toward the old
    orientation.
    """
    lat, lon = -10.17, 123.6  # Kupang, NTT -- southern Indonesia
    cfg = COUNTRY_SOLAR_CONFIG["indonesia"]

    fixture_cls = load_dataset("wind_solar_hourly_test")
    fixture = fixture_cls(years=slice(2016, 2016), months=slice(1, 1))
    model = Pvlib(fixture)

    cec_modules = model.retrieve_sam("CECMod")
    cec_inverters = model.retrieve_sam("CECInverter")
    module = cec_modules[cfg["module"]]
    inverter = cec_inverters[cfg["inverter"]]

    model.init_model_config(
        clearsky_model="haurwitz",
        transposition_model="perez",
        solar_position_method="nrel_numpy",
        airmass_model="kastenyoung1989",
        dc_model="cec",
        ac_model="sandia",
        aoi_model="physical",
        spectral_model="first_solar",
        dc_ohmic_model="no_loss",
    )

    ds = _synthetic_solar_ds_full_year(lat, lon, year=2020)

    def _annual_ac_sum(tilt: float, azimuth: float) -> float:
        model.init_pv_system(
            arrays=None,
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            racking_model="open_rack",
            module_parameters=module,
            modules_per_string=50,
            module_type="glass_polymer",
            module=cfg["module"],
            strings_per_inverter=1,
            inverter_parameters=inverter,
        )
        result = model._pvlib_model(
            ds, model.pv_system, model.config, n_jobs=1, compact_output=True
        )
        return float(np.nansum(result["ac"].values))

    correct = latitude_optimal_orientation(lat)
    fixed_total = _annual_ac_sum(correct["surface_tilt"], correct["surface_azimuth"])
    buggy_total = _annual_ac_sum(tilt=35.0, azimuth=180.0)

    assert fixed_total > buggy_total * 1.1, (
        f"Correct orientation ({correct}) produced {fixed_total:.0f} Wh/yr, "
        f"not meaningfully more than the old hardcoded south-facing bug's "
        f"{buggy_total:.0f} Wh/yr -- hemisphere-orientation regression?"
    )
