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

"""Unit tests for hemisphere-aware panel orientation.

Guards against a regression where the pvlib model (or its callers) assume
a fixed, south-facing (``surface_azimuth=180``) orientation regardless of
hemisphere -- which is physically backwards south of the equator (e.g.
Indonesia's NTT region).
"""

import pytest

from geodata.model.pvlib import latitude_optimal_orientation


def test_northern_hemisphere_faces_south():
    orientation = latitude_optimal_orientation(35.0)
    assert orientation["surface_azimuth"] == 180.0
    assert orientation["surface_tilt"] > 0


def test_southern_hemisphere_faces_north():
    """South of the equator, an equator-facing panel points north, not south."""
    orientation = latitude_optimal_orientation(-8.5)  # e.g. NTT, Indonesia
    assert orientation["surface_azimuth"] == 0.0
    assert orientation["surface_tilt"] > 0


def test_equator_defaults_to_northern_convention():
    orientation = latitude_optimal_orientation(0.0)
    assert orientation["surface_azimuth"] == 180.0
    assert orientation["surface_tilt"] == 0.0


@pytest.mark.parametrize(
    "lat_pos,lat_neg",
    [(5.0, -5.0), (24.9, -24.9), (25.0, -25.0), (49.0, -49.0), (60.0, -60.0)],
)
def test_tilt_symmetric_across_equator(lat_pos, lat_neg):
    """Tilt magnitude should only depend on |latitude|, only azimuth flips."""
    pos = latitude_optimal_orientation(lat_pos)
    neg = latitude_optimal_orientation(lat_neg)
    assert pos["surface_tilt"] == pytest.approx(neg["surface_tilt"])
    assert pos["surface_azimuth"] == 180.0
    assert neg["surface_azimuth"] == 0.0


def test_high_latitude_caps_at_40_degrees():
    orientation = latitude_optimal_orientation(65.0)
    assert orientation["surface_tilt"] == 40.0
