# Copyright 2025 Michael Davidson (UCSD), Xiqiang Liu (UCSD)

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

import logging
import multiprocessing as mp
import os

import pandas as pd
import xarray as xr
from herbie import FastHerbie, Herbie

from ....logging import redirect_stdout_to_logger
from ..._base import AtomicDataset
from .._base import HRRRBaseDataset

logger = logging.getLogger(__name__)


class HRRR3DWindHourlyDataset(HRRRBaseDataset):
    """HRRR3DWindHourlyDataset is a class that encaps a dataset from the HRRR
    dataset. It provides a streamlined workflow for downloading, preprocessing,
    and storing of these datasets.

    Variables:
        - u: zonal wind component
        - v: meridional wind component
        - gh: geopotential height

    Important Coordinates:
        - time: time of the observation
        - level: hybrid pressure level
        - y: latitude
        - x: longitude
    """

    weather_config = "hrrr_wind_3d_hourly"
    product = "nat"  # Use "nat" product for 3D data

    def _download_file(self, file: AtomicDataset):
        year, month, day = file.year, file.month, file.day

        date_range = pd.date_range(
            f"{year}-{month}-{day}",
            f"{year}-{month}-{day+1}",
            freq="h",
            inclusive="left",
        )

        fh = FastHerbie(
            date_range,
            model=self.module,
            product=self.product,
            max_threads=mp.cpu_count() * 2,
            save_dir=self._herbie_save_dir.name,
            priority=self._priority,
        )

        with redirect_stdout_to_logger(logger, logging.INFO):
            logger.info(f"Downloading HRRR wind data in bulk for {year}/{month}")
            fh.download(":[UV]GRD:[1,8]0 m")
            fh.download(":[UV]GRD:[1234] hybrid level")
            fh.download(":HGT:[1234] hybrid level")

            uv_10 = []
            uv_80 = []
            uv_hybrid = []
            hgt_hybrid = []

            for hour in date_range:
                h = Herbie(
                    hour,
                    model=self.module,
                    product=self.product,
                    save_dir=self._herbie_save_dir.name,
                    priority=self._priority,
                )

                try:
                    uv_10.append(
                        h.xarray(":[UV]GRD:10 m", remove_grib=False).rename(
                            {"u10": "u", "v10": "v"}
                        )
                    )
                    uv_80.append(h.xarray(":[UV]GRD:80 m", remove_grib=False))
                    uv_hybrid.append(
                        h.xarray(":[UV]GRD:[1234] hybrid level", remove_grib=False)
                    )
                    hgt_hybrid.append(
                        h.xarray(":HGT:[1234] hybrid level", remove_grib=False)
                    )
                except ValueError:
                    logger.warning(f"No data found for {hour}, skipping.")

            # Offload to disk temporarily to save memory
            xr.concat(uv_10, dim="time").to_netcdf(
                os.path.join(self._herbie_save_dir.name, "uv_10.nc")
            )
            del uv_10
            xr.concat(uv_80, dim="time").to_netcdf(
                os.path.join(self._herbie_save_dir.name, "uv_80.nc")
            )
            del uv_80
            xr.concat(uv_hybrid, dim="time").to_netcdf(
                os.path.join(self._herbie_save_dir.name, "uv_hybrid.nc")
            )
            del uv_hybrid
            xr.concat(hgt_hybrid, dim="time").to_netcdf(
                os.path.join(self._herbie_save_dir.name, "hgt_hybrid.nc")
            )
            del hgt_hybrid

            uv_10 = xr.open_dataset(
                os.path.join(self._herbie_save_dir.name, "uv_10.nc"), chunks="auto"
            )
            uv_80 = xr.open_dataset(
                os.path.join(self._herbie_save_dir.name, "uv_80.nc"), chunks="auto"
            )
            uv_hybrid = xr.open_dataset(
                os.path.join(self._herbie_save_dir.name, "uv_hybrid.nc"), chunks="auto"
            )
            hgt_hybrid = xr.open_dataset(
                os.path.join(self._herbie_save_dir.name, "hgt_hybrid.nc"), chunks="auto"
            )

            ds: xr.Dataset = xr.concat([uv_10, uv_80], dim="heightAboveGround")
            heights = ds["heightAboveGround"].broadcast_like(ds["u"])

            ds["u"] = xr.concat(
                [ds["u"].rename({"heightAboveGround": "hybrid"}), uv_hybrid["u"]],
                dim="hybrid",
            )
            ds["v"] = xr.concat(
                [ds["v"].rename({"heightAboveGround": "hybrid"}), uv_hybrid["v"]],
                dim="hybrid",
            )
            del ds["heightAboveGround"]

            ds["gh"] = xr.concat(
                [heights.rename({"heightAboveGround": "hybrid"}), hgt_hybrid["gh"]],
                dim="hybrid",
            ).astype("float32")
            ds = ds.rename({"hybrid": "level"}).sortby("level")

            ds["level"].values[-2:] = [-1, -2]
            ds["level"] = ds["level"].astype("int8")

            ds = ds.sortby("level")

            try:
                del ds.attrs["search"]
                del ds.attrs["local_grib"]
                del ds.attrs["remote_grib"]
            except KeyError:
                pass

            ds.to_netcdf(file.path)

        # NOTE: Flush temporary FastHerbie save directory to save space, since we no
        # longer need the raw downloaded files
        self._herbie_save_dir.cleanup()
