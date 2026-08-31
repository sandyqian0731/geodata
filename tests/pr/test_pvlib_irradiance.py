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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Unit tests for the pvlib model's irradiance preparation.

ERA5's ``fdir`` (geodata's ``influx_direct``) is the direct irradiance on a
HORIZONTAL plane (BHI), not the direct normal irradiance (DNI) that
``pvlib.modelchain.ModelChain`` expects. These tests pin the two conversions:

- DNI = BHI / cos(zenith), zeroed at grazing/night zenith angles, instead of
  the historical rename of ``influx_direct`` straight to ``dni`` (which
  understated DNI by cos(zenith));
- GHI = DHI + BHI, both horizontal fluxes summed directly, instead of the
  historical DHI + BHI * cos(zenith) (which double-applied the projection);
  ERA5's ``ssrd`` is used verbatim when the dataset still carries it.

Offline: synthetic single-cell datasets and synthetic zenith angles only.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geodata.datasets import load_dataset
from geodata.model.pvlib import Pvlib
from geodata.model.pvlib.calculations import (
    calculate_dni,
    calculate_ghi,
    calculate_pvlib_solarposition,
)


def _single_cell_ds(nt: int = 1, lat: float = 34.3, lon: float = 108.9, **data_vars):
    """A (time, y, x) = (nt, 1, 1) dataset with the given per-time values."""
    time = pd.date_range("2020-06-21 04:00", periods=nt, freq="h", tz=None)
    variables = {
        name: (["time", "y", "x"], np.asarray(values, dtype=float).reshape(nt, 1, 1))
        for name, values in data_vars.items()
    }
    return xr.Dataset(variables, coords={"time": time, "y": [lat], "x": [lon]})


def test_dni_is_bhi_over_cos_zenith():
    """zenith = 60 degrees, fdir = 400 W/m2 -> DNI = 400 / cos(60) = 800."""
    ds = _single_cell_ds(influx_direct=[400.0])
    zenith = pd.Series([60.0])
    dni = calculate_dni(ds, zenith)
    assert dni.values.shape == (1, 1, 1)
    assert dni.values.ravel()[0] == pytest.approx(800.0)


def test_dni_zero_at_grazing_and_night_zenith():
    """DNI is 0 at and beyond the 88-degree zenith cutoff (incl. night)."""
    ds = _single_cell_ds(nt=3, influx_direct=[400.0, 50.0, 10.0])
    zenith = pd.Series([88.0, 90.0, 120.0])
    dni = calculate_dni(ds, zenith)
    assert (dni.values == 0).all(), f"expected all zeros, got {dni.values.ravel()}"


def test_dni_clips_negative_input_to_zero():
    """ERA5 numerical noise (tiny negative fdir) must not produce negative DNI."""
    ds = _single_cell_ds(influx_direct=[-0.5])
    zenith = pd.Series([45.0])
    dni = calculate_dni(ds, zenith)
    assert dni.values.ravel()[0] == 0.0


def test_ghi_is_dhi_plus_bhi_without_zenith_projection():
    """GHI = DHI + BHI = 100 + 400 = 500; both inputs are already horizontal.

    The historical DHI + BHI * cos(zenith) gave 300 at zenith = 60 degrees.
    """
    ds = _single_cell_ds(influx_diffuse=[100.0], influx_direct=[400.0])
    zenith = pd.Series([60.0])
    ghi = calculate_ghi(ds, zenith)
    assert ghi.values.ravel()[0] == pytest.approx(500.0)


def test_ghi_uses_ssrd_when_present():
    """ERA5's ssrd IS GHI by definition; use it verbatim when available."""
    ds = _single_cell_ds(
        influx_diffuse=[100.0], influx_direct=[400.0], ssrd=[555.0]
    )
    zenith = pd.Series([60.0])
    ghi = calculate_ghi(ds, zenith)
    assert ghi.values.ravel()[0] == pytest.approx(555.0)


def test_prepare_pvlib_ds_derives_dni_from_real_solar_position():
    """The prepared weather dataset must carry derived DNI, not renamed fdir.

    Uses the real solar-position path on a synthetic daytime dataset:
    everywhere the sun is up (zenith < 88 degrees) and fdir > 0, the prepared
    ``dni`` must exceed ``influx_direct`` (1/cos(zenith) > 1 off-zenith), and
    ``ghi`` must equal ``dhi + influx_direct`` exactly.
    """
    fixture_cls = load_dataset("wind_solar_hourly_test")
    fixture = fixture_cls(years=slice(2016, 2016), months=slice(1, 1))
    model = Pvlib(fixture)

    nt = 24
    ds = _single_cell_ds(
        nt=nt,
        influx_diffuse=np.full(nt, 100.0),
        influx_direct=np.full(nt, 400.0),
        dewpoint_temperature=np.full(nt, 280.0),
        temperature=np.full(nt, 295.0),
        wnd100m=np.full(nt, 5.0),
    )

    prepared = model._prepare_pvlib_ds(
        ds,
        "influx_diffuse",
        "influx_direct",
        "dewpoint_temperature",
        "temperature",
        "wnd100m",
    )

    assert "dni" in prepared.data_vars and "ghi" in prepared.data_vars

    zenith = np.asarray(calculate_pvlib_solarposition(ds)["zenith"].values)
    dni = prepared["dni"].values.ravel()
    ghi = prepared["ghi"].values.ravel()
    dhi = prepared["dhi"].values.ravel()
    bhi = ds["influx_direct"].values.ravel()

    sun_up = zenith < 88.0
    assert sun_up.any() and (~sun_up).any(), "test span should include day and night"

    expected_dni = np.zeros_like(bhi)
    expected_dni[sun_up] = bhi[sun_up] / np.cos(np.deg2rad(zenith[sun_up]))
    np.testing.assert_allclose(dni, expected_dni)

    # Off-zenith, DNI must exceed the horizontal beam component.
    assert (dni[sun_up] > bhi[sun_up]).all()
    # At/below the cutoff, DNI is exactly 0 -- no 1/cos blow-up at night.
    assert (dni[~sun_up] == 0).all()

    np.testing.assert_allclose(ghi, dhi + bhi)
