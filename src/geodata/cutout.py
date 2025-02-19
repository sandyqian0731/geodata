# Copyright 2016-2017 Gorm Andresen (Aarhus University), Jonas Hoersch (FIAS), Tom Brown (FIAS)
# Copyright 2020 Michael Davidson (UCSD), William Honaker, Jiahe Feng (UCSD), Yuanbo Shi
# Copyright 2023-2025 Xiqiang Liu

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


"""
Cutout class to handle a subset of a Dataset.
"""

import logging
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import numpy as np
import pyproj
import shapely
import xarray as xr
from shapely.geometry import box
from tqdm.auto import tqdm

from . import config
from .convert import (
    convert_heat_demand,
    convert_pm25,
    convert_pv,
    convert_solar_thermal,
    convert_wind,
    convert_windspd,
    convert_windwpd,
    get_orientation,
)
from .datasets._base import BaseDataset
from .mask import Mask
from .preparation import (
    cutout_get_meta,
    cutout_get_meta_view,
    cutout_prepare,
    cutout_produce_specific_dataseries,
)
from .resource import (
    get_solarpanelconfig,
    get_windturbineconfig,
    windturbine_smooth,
)

logger = logging.getLogger(__name__)


class Cutout:
    """Cutout class to handle a subset of a Dataset.

    Args:
        name (str): Name of the cutout. This name will be used to uniquely
            identify the cutout. If a cutout with the same name already exists,
            the existing cutout will be retrieved.
        dataset_cls (type[BaseDataset]): Dataset class to use for the cutout.
        years (slice): years of the cutout.
        cutout_dir (str): path to the cutout directory. This is optional. If not
            specified, the cutout will be stored in the default cutout directory
            under `GEODATA_ROOT`.
        bounds (Optional[Sequence]): bounds of the cutout. Optional. If not specified,
            the bounds will be automatically generated.
        months (Optional[slice]): months of the cutout. Optional. If not specified,
            the months will be automatically generated.
        xs (Optional[slice]): longitude coordinates of the cutout. Optional. If not specified,
            the x coordinates will be automatically generated.
        ys (Optional[slice]): latitude coordinates of the cutout. Optional. If not specified,
            the y coordinates will be automatically generated.
    """

    def __init__(
        self,
        name: str,
        dataset_cls: type[BaseDataset],
        years: slice,
        cutout_dir: Union[str, Path] = config.cutout_dir,
        bounds: Optional[Sequence] = None,
        months: Optional[slice] = None,
        xs: Optional[slice] = None,
        ys: Optional[slice] = None,
    ):
        self.name = name
        self.cutout_dir = Path(cutout_dir, name)

        self.dataset_cls = dataset_cls

        self.prepared = False
        self.empty = False
        self.meta = None
        self.merged_mask = None
        self.shape_mask = None
        self.area = None

        params_dict = {
            "years": years,
            "months": months,
            "xs": xs,
            "ys": ys,
        }

        if bounds is not None and (xs is not None or ys is not None):
            raise TypeError("Cannot specify both bounds and xs/ys arguments.")

        if bounds is not None:
            # if passed bounds array instead of xs, ys slices
            x1, y1, x2, y2 = bounds
            params_dict.update(xs=slice(x1, x2), ys=slice(y1, y2))

        if months is None:
            logger.info("No months specified, defaulting to 1-12")
            params_dict.update(months=slice(1, 12))

        if self.cutout_dir.is_dir():
            # Check if cutout directory exists
            # If cutout dir exists, check completness of files
            if self.meta_path.is_file():  # open existing meta file
                self.meta = xr.open_dataset(self.get_filename()).stack(
                    dim={"year-month": ("year", "month")}
                )

            if all(self.catalog):
                logger.info("All cutout (%s, %s) files available.", name, cutout_dir)

                # At least one of xs, ys is in params_dict
                if {"xs", "ys"}.intersection(params_dict):
                    # Passed some subsetting bounds
                    self.meta = self.get_meta_view(**params_dict)
                    if self.meta is not None:
                        # Subset is available
                        self.prepared = True
                        logger.info("Cutout subset prepared: %s", self)
                    else:
                        logger.info("Cutout subset not available: %s", self)
                else:
                    # No subsetting of bounds. Keep full cutout
                    self.prepared = True
                    logger.info("Cutout prepared: %s", self)

            else:
                #   Not all files accounted for
                self.prepared = False
                logger.info("Cutout (%s, %s) not complete.", name, cutout_dir)

        # In case
        if not self.prepared:
            logger.info("Cutout (%s, %s) not found or incomplete.", name, cutout_dir)

            if {"xs", "ys", "years"}.difference(params_dict):
                raise TypeError(
                    "Arguments `xs`, `ys`, and `years` need to be specified to create "
                    "a cutout."
                )

            if self.meta is not None:
                # if meta.nc exists, close and delete it
                self.meta.close()
                self.meta_path.unlink()

            self.meta = self.get_meta(**params_dict)
            self.cutout_dir.mkdir(parents=True, exist_ok=True)

            # Write meta file
            self.meta_clean.unstack("year-month").to_netcdf(self.meta_path)

    def _get_filename(self, year: int | None = None, month: int | None = None):
        """Return path to dataset xarray files related to this Cutout.
        If both year and month are None, return path to meta.nc file.

        Args:
            year (int): The year to get the filename for. Defaults to None.
            month (int): The month to get the filename for. Defaults to None.

        Returns:
            Path: Path to the dataset xarray file.
        """

        if year is None and month is None:
            return self.cutout_dir / "meta.nc"

        if year is not None:
            if month is not None:
                return self.cutout_dir / f"{year}{month:0>2}.nc"

        return self.cutout_dir / f"{year}.nc"

    @property
    def catalog(self):
        """A generator that yields all dataset files."""

        for year in self.coords["year"]:
            for month in self.coords["month"]:
                yield self._get_filename(year, month)

    @property
    def meta_path(self):
        """Path to the metadata file."""
        return self._get_filename()

    @property
    def meta_data_config(self):
        """Metadata configuration for the Cutout"""

        return {
            "tasks_func": self.dataset_cls.tasks_func,
            "prepare_func": self.dataset_cls.meta_prepare_func,
            "file_granularity": self.dataset_cls.frequency,
        }

    @property
    def weather_config(self):
        """The weather data configuration for the Cutout."""

        return self.dataset_cls.weather_config

    @property
    def info(self):
        """Summary information about the Cutout."""
        return {
            "name": self.name,
            "prepared": self.prepared,
            "shape": self.shape,
            "extent": self.extent,
            "years": self.years,
            "months": self.months,
            "meta": self.meta,
        }

    @property
    def projection(self):
        """The projection of the Cutout."""
        return self.dataset_cls.projection

    @property
    def coords(self):
        """The coordinates covered by the Cutout."""
        return self.meta.coords

    @property
    def meta_clean(self):
        # produce clean version of meta file for export to NetCDF (cannot export slices)
        meta = self.meta
        if meta.attrs.get("view", {}):
            view = {}
            for name, value in meta.attrs["view"].items():
                view.update({name: [value.start, value.stop]})
            meta.attrs["view"] = str(view)
        return meta

    @property
    def shape(self):
        """The shape of the Cutout by (y, x)."""
        return len(self.coords["y"]), len(self.coords["x"])

    @property
    def extent(self):
        """The extent of the Cutout by (x_min, x_max, y_min, y_max)."""
        return list(self.coords["x"].values[[0, -1]]) + list(
            self.coords["y"].values[[-1, 0]]
        )

    @property
    def years(self):
        """Cutout's covered years as slice object."""
        return slice(*self.coords["year"].values)

    @property
    def months(self):
        """Cutout's covered months as slice object."""
        return slice(*self.coords["month"].values)

    def grid_coordinates(self):
        """Return grid coordinates of the Cutout."""
        xs, ys = np.meshgrid(self.coords["x"], self.coords["y"])
        return np.asarray((np.ravel(xs), np.ravel(ys))).T

    def grid_cells(self):
        """Return grid cells of the Cutout."""
        coords = self.grid_coordinates()
        span = (coords[self.shape[1] + 1] - coords[0]) / 2
        return [box(*c) for c in np.hstack((coords - span, coords + span))]

    def __repr__(self):
        yearmonths = self.coords["year-month"].to_index()
        return "<Cutout {} x={:.2f}-{:.2f} y={:.2f}-{:.2f} time={}/{}-{}/{} {}prepared>".format(
            self.name,
            self.coords["x"].values[0],
            self.coords["x"].values[-1],
            self.coords["y"].values[0],
            self.coords["y"].values[-1],
            yearmonths[0][0],
            yearmonths[0][1],
            yearmonths[-1][0],
            yearmonths[-1][1],
            "" if self.prepared else "UN",
        )

    def add_mask(self, name: str, merged_mask: bool = True, shape_mask: bool = True):
        """Add mask attribute to the cutout, from a previously saved mask objects.
        The masks will be coarsened to the same dimension with the cutout metadata in xarray.

        Args:
            name (str): The name of the previously saved mask. The mask object should be saved in
                mask_dir in config.py.
            merged_mask (bool): If true, the program will try to include the merged_mask from the mask object.
                Defaults to True.
            shape_mask (bool): If true, the program will try to include the extracted dictionary of
                shape_mask from the mask object. Defaults to True.
        """
        # make sure data is in correct format for coarsening
        xr_ds = ds_reformat_index(self.meta)
        mask = Mask.from_name(name, mask_dir=config.MASK_DIR)

        if not mask.merged_mask and not mask.shape_mask:
            raise ValueError(
                f"No mask found in {mask.name}. Please create a proper mask object first."
            )

        if mask.merged_mask and merged_mask:
            self.merged_mask = coarsen(mask.load_merged_xr(), xr_ds)
            logger.info("Cutout.merged_mask added.")

        if mask.shape_mask and shape_mask:
            self.shape_mask = {
                k: coarsen(v, xr_ds) for k, v in mask.load_shape_xr().items()
            }
            logger.info("Cutout.shape_mask added.")

    def add_grid_area(
        self, axis: tuple[str] = ("lat", "lon"), adjust_coord: bool = True
    ):
        """Add attribute 'area' to the cutout containing area for each grid cell
        in the cutout metedata xarray.

        Args:
            axis (tuple[str]): The name of the axes to include in the result xarray dataset.
                Defaults to ("lat", "lon").
            adjust_coord (bool): Whether to sort the data by latitude and longitude values if true.
                Defaults to True.
        """
        xr_ds = ds_reformat_index(self.meta)
        area_arr = np.zeros((xr_ds.lat.shape[0], xr_ds.lon.shape[0]))
        lat_diff = np.abs((xr_ds.lat[1].values - xr_ds.lat[0].values))
        for i, lat in enumerate(xr_ds.lat.values):
            lat_bottom = lat - lat_diff / 2
            lat_top = lat + lat_diff / 2
            # calculate the area for grid cells with same latitude
            area_arr[i] = np.round(
                calc_grid_area(
                    [
                        (xr_ds.lon.values[0], lat_top),
                        (xr_ds.lon.values[0], lat_bottom),
                        (xr_ds.lon.values[1], lat_bottom),
                        (xr_ds.lon.values[1], lat_top),
                    ]
                ),
                2,
            )
        if axis == ("lat", "lon"):
            xr_ds = xr_ds.assign({"area": (axis, area_arr)})
        elif axis == ("time", "lat", "lon"):
            xr_ds = xr_ds.assign({"area": (axis, [area_arr])})

        if adjust_coord:
            if (
                ("lon" not in xr_ds.coords)
                and ("lat" not in xr_ds.coords)
                and ("x" in xr_ds.coords and "y" in xr_ds.coords)
            ):
                xr_ds = xr_ds.rename({"x": "lon", "y": "lat"})
            xr_ds = xr_ds.sortby(["lat", "lon"])

        self.area = xr_ds

    def mask(
        self,
        dataset: xr.Dataset,
        true_area: bool = True,
        merged_mask: bool = True,
        shape_mask: bool = True,
    ):
        """Mask a converted `xarray.Dataset` from cutout with previously added mask attribute
        with optional area inclusion, and return a dictionary of xarray Dataset.

        The program will search for 'merged_mask' and 'shape_mask' attributes in the
        cutout object, these Xarray data can be generate through 'add_mask', unless the user
        specify 'merged_mask = False' or 'shape_mask = False', the masks in shape_mask
        will have the same key in the dictionary returned, and the mask for merged_mask will
        have the key name "merged_mask".

        Args:
            dataset (xr.Dataset): The dataset to be masked.
            true_area (bool): Whether the returned masks will have the area variable. Defaults to True.
            merged_mask (bool): If true, the program will try to
                include the merged_mask from the cutout object. Defaults to True.
            shape_mask (bool): If true, the program will try to
                include the extracted dictionary of shape_mask from the cutout object. Defaults to True.

        Returns:
            dict[xr.Dataset]: A dictionary of xarray Dataset with masks combined with the dataset.
        """
        axis = ("lat", "lon")

        dataset = ds_reformat_index(dataset)

        dataset = dataset.transpose("time", "lat", "lon")

        if self.merged_mask is None and self.shape_mask is None:
            raise ValueError(
                "No mask found in cutout. Please add masks with self.add_mask()"
            )

        if self.area is None and true_area:
            raise ValueError(
                "No area data found. Please call self.add_grid_area() or set true_area to False."
            )

        res = {}

        if self.merged_mask is not None and merged_mask:
            ds = dataset.assign({"mask": (axis, self.merged_mask.data[0])})
            if true_area:
                ds = ds.assign({"area": (axis, self.area["area"].data)})
            res["merged_mask"] = ds
            logger.info("merged_mask combined with dataset. ")

        if self.shape_mask is not None and shape_mask:
            for key, val in self.shape_mask.items():
                ds = dataset.assign({"mask": (axis, val.data[0])})
                if true_area:
                    ds = ds.assign({"area": (axis, self.area["area"].data)})
                res[key] = ds
        logger.info("shape_mask combined with dataset. ")

        return res

    # Preparation functions
    get_meta = cutout_get_meta
    get_meta_view = cutout_get_meta_view  # preparation.cutout_get_meta_view
    prepare = cutout_prepare  # preparation.cutout_prepare
    produce_specific_dataseries = cutout_produce_specific_dataseries

    # Conversion and aggregation functions
    def _convert_cutout(
        self, convert_func: callable, show_progress: bool = False, **convert_kwds
    ) -> xr.DataArray:
        """Convert and aggregate a weather-based renewable generation time-series.

        NOTE: This is a function designed to be used internally instead of being used by
        the user themself. Instead, this is a gateway function that is called by all
        the individual time-series generation functions like pv and wind. Thus, all its parameters are also
        available from these.

        Args:
            convert_func (callable): Function to convert the cutout
            show_progress (bool): Show progress bar. Set to False by default.
            **convert_kwds: Keyword arguments passed to convert_func

        Returns:
            xr.DataArray: Converted cutout
        """

        if not self.prepared:
            raise RuntimeError("The cutout has to be prepared first.")

        results = []
        yearmonths = self.coords["year-month"].to_index()

        if isinstance(show_progress, str):
            prefix = show_progress
        else:
            func_name = (
                convert_func.__name__[len("convert_") :]
                if convert_func.__name__.startswith("convert_")
                else convert_func.__name__
            )
            prefix = f"Convert {func_name}"

        for ym in tqdm(
            yearmonths, desc=prefix, disable=not show_progress, dynamic_ncols=True
        ):
            with xr.open_dataset(self._get_filename(ym)) as ds:
                if "view" in self.meta.attrs:
                    if isinstance(self.meta.attrs["view"], str):
                        self.meta.attrs["view"] = {}
                        self.meta.attrs.setdefault("view", {})["x"] = slice(
                            min(self.meta.coords["x"]).values.tolist(),
                            max(self.meta.coords["x"]).values.tolist(),
                        )
                        self.meta.attrs.setdefault("view", {})["y"] = slice(
                            min(self.meta.coords["y"]).values.tolist(),
                            max(self.meta.coords["y"]).values.tolist(),
                        )
                    ds = ds.sel(**self.meta.attrs["view"])

                da = convert_func(ds, **convert_kwds)
                results.append(da.load())

        return xr.concat(results, dim="time")

    def heat_demand(
        self,
        threshold: float = 15.0,
        a: float = 1.0,
        constant: float = 0.0,
        hour_shift: float = 0.0,
        **params,
    ):
        """Convert outside temperature into daily heat demand using the
        degree-day approximation.

        Since "daily average temperature" means different things in
        different time zones and since xarray coordinates do not handle
        time zones gracefully like pd.DateTimeIndex, you can provide an
        hour_shift to redefine when the day starts.

        E.g. for Moscow in winter, hour_shift = 4, for New York in winter,
        hour_shift = -5

        This time shift applies across the entire spatial scope of ds for
        all times. More fine-grained control will be built in a some
        point, i.e. space- and time-dependent time zones.

        WARNING: Because the original data is provided every month, at the
        month boundaries there is untidiness if you use a time shift. The
        resulting xarray will have duplicates in the index for the parts
        of the day in each month at the boundary. You will have to
        re-average these based on the number of hours in each month for
        the duplicated day.

        Args:
            threshold (float): Outside temperature in degrees Celsius above which there is no heat demand.
            a (float): Linear factor relating heat demand to outside temperature.
            constant (float): Constant part of heat demand that does not depend on outside
                temperature (e.g. due to water heating).
            hour_shift (float): Time shift relative to UTC for taking daily average

        Returns:
            xr.DataArray: Heat demand

        Note:
            You can also specify all of the general conversion arguments
            documented in the `convert_cutout` function.
        """

        return self._convert_cutout(
            convert_func=convert_heat_demand,
            threshold=threshold,
            a=a,
            constant=constant,
            hour_shift=hour_shift,
            **params,
        )

    def temperature(self, **convert_params):
        """Convert temperature in Cutout to outside temperature.

        Args:
            convert_params: Keyword arguments passed to `convert_cutout` function

        Returns:
            xr.DataArray: Data of the Cutout with temperature converted to outside temperatures.
        """
        return self._convert_cutout(
            convert_func=lambda ds: ds["temperature"] - 273.15, **convert_params
        )

    def soil_temperature(self, **convert_params):
        """Return soil temperature (useful for e.g. heat pump T-dependent
        coefficient of performance).

        Args:
            convert_params: Keyword arguments passed to `convert_cutout` function

        Returns:
            xr.DataArray: Data of the Cutout with temperature converted to soil temperatures.
        """
        return self._convert_cutout(
            convert_func=lambda ds: (ds["soil temperature"] - 273.15).fillna(0.0),
            **convert_params,
        )

    def solar_thermal(
        self,
        orientation: Optional[Union[dict, str, callable]] = None,
        trigon_model: str = "simple",
        clearsky_model: Literal["simple", "enhanced"] = "simple",
        c0: float = 0.8,
        c1: float = 3.0,
        t_store: float = 80.0,
        **params,
    ):
        """Convert downward short-wave radiation flux and outside temperature
        into time series for solar thermal collectors.

        Mathematical model and defaults for c0, c1 based on model in [1].

        Args:
            orientation (Union[dict, str, callable]): Panel orientation with slope and azimuth
                (units of degrees), or 'latitude_optimal'.
            trigon_model (str): Type of trigonometry model
            clearsky_model (str): Type of clearsky model for diffuse irradiation. Either
                `simple` or `enhanced`.
            c0 (float): Parameter for model in [1] This defaults to 0.8.
            c1 (float): Parameter for model in [1] This defaults to 3.0.
            t_store (float): Store temperature in degree Celsius

        Note:
            You can also specify all of the general conversion arguments
            documented in the `convert_cutout` function.

        References:
            [1] Henning and Palzer, Renewable and Sustainable Energy Reviews 30
            (2014) 1003-1018
        """

        if orientation is None:
            orientation = {"slope": 45.0, "azimuth": 180.0}

        if not callable(orientation):
            orientation = get_orientation(orientation)

        return self._convert_cutout(
            convert_func=convert_solar_thermal,
            orientation=orientation,
            trigon_model=trigon_model,
            clearsky_model=clearsky_model,
            c0=c0,
            c1=c1,
            t_store=t_store,
            **params,
        )

    def wind(
        self,
        turbine: Union[str, dict],
        method: Literal["simple", "interpolation", "extrapolation"],
        smooth: Union[bool, dict] = False,
        **params,
    ):
        """Convert wind speed time-series into wind generation time-series.

        Args:
            turbine (Union[str, dict]): Name of a turbine or a dictionary with the parameters
                for the wind turbine in [2].
            smooth (Union[bool, dict]): If True, the wind speed time-series will be smoothed
                before conversion. If False, no smoothing will be applied. If a dictionary is
                passed, the smoothing parameters will be used.
            **params: Keyword arguments passed to `convert_cutout` function
        """

        if isinstance(turbine, str):
            turbine = get_windturbineconfig(turbine)

        if smooth:
            turbine = windturbine_smooth(turbine, params=smooth)

        match method:
            case "simple":
                return self._convert_cutout(
                    convert_func=convert_wind, turbine=turbine, **params
                )

            case _:
                raise ValueError(f"Method {method} not supported.")

    def windspd(self, **params):
        """
        Generate wind speed time-series

        convert.convert_cutout → convert.convert_windspd

        Parameters
        ----------
        **params
            Must have 1 of:
                turbine : str or dict
                        Name of a turbine
                hub_height : num
                        Extrapolation height

            Can also specify all of the general conversion arguments
            documented in the `convert_cutout` function.
                e.g. var_height='lml'

        """

        if "turbine" in params:
            turbine = params.pop("turbine")
            if isinstance(turbine, str):
                turbine = get_windturbineconfig(turbine)
            else:
                raise ValueError(f"Turbine ({turbine}) not found.")
            hub_height = itemgetter("hub_height")(turbine)
        elif "hub_height" in params:
            hub_height = params.pop("hub_height")
        elif "to_height" in params:
            hub_height = params.pop("to_height")
        else:
            raise ValueError("Either a turbine or hub_height must be specified.")

        params["hub_height"] = hub_height

        return self._convert_cutout(convert_func=convert_windspd, **params)

    def windwpd(self, **params):
        """
        Generate wind power density time-series

        convert.convert_cutout → convert.convert_windwpd

        Parameters
        ----------
        **params
                Must have 1 of:
                        turbine : str or dict
                                Name of a turbine
                        hub_height : num
                                Extrapolation height

                Can also specify all of the general conversion arguments
                documented in the `convert_cutout` function.
                        e.g. var_height='lml'

        """

        if "turbine" in params:
            turbine = params.pop("turbine")
            if isinstance(turbine, str):
                turbine = get_windturbineconfig(turbine)
            else:
                raise ValueError(f"Turbine ({turbine}) not found.")
            hub_height = itemgetter("hub_height")(turbine)
        elif "hub_height" in params:
            hub_height = params.pop("hub_height")
        elif "to_height" in params:
            hub_height = params.pop("to_height")
        else:
            raise ValueError("Either a turbine or hub_height must be specified.")

        params["hub_height"] = hub_height

        return self._convert_cutout(convert_func=convert_windwpd, **params)

    def pv(
        self,
        panel: Union[str, dict],
        orientation: Union[str, dict, callable],
        clearsky_model: Optional[str] = None,
        **params,
    ):
        """Convert downward-shortwave, upward-shortwave radiation flux and
        ambient temperature into a pv generation time-series.

        Args:
            panel (Union[str, dict]): Panel name known to the reatlas client or a panel config
                dictionary with the parameters for the electrical model in [3].
            orientation (Union[str, dict, callback]): Panel orientation can be chosen from either
                'latitude_optimal', a constant orientation {'slope': 0.0,
                'azimuth': 0.0} or a callback function with the same signature
                as the callbacks generated by the
                `geodata.pv.orientation.make_*` functions.
            clearsky_model (Optional[str]): Either the 'simple' or the 'enhanced' Reindl clearsky
                model. The default choice of None will choose dependending on
                data availability, since the 'enhanced' model also
                incorporates ambient air temperature and relative humidity.

        Returns:
            xr.DataArray: Time-series or capacity factors based on additional general
            conversion arguments.

        Note:
            You can also specify all of the general conversion arguments
            documented in the `convert_cutout` function.

        References:
            [1] Soteris A. Kalogirou. Solar Energy Engineering: Processes and Systems,
            pages 49-117,469-516. Academic Press, 2009. ISBN 0123745012.
            [2] D.T. Reindl, W.A. Beckman, and J.A. Duffie. Diffuse fraction correla-
            tions. Solar Energy, 45(1):1 - 7, 1990.
            [3] Hans Georg Beyer, Gerd Heilscher and Stefan Bofinger. A Robust Model
            for the MPP Performance of Different Types of PV-Modules Applied for
            the Performance Check of Grid Connected Systems, Freiburg, June 2004.
            Eurosun (ISES Europe Solar Congress).
        """

        if isinstance(panel, str):
            panel = get_solarpanelconfig(panel)
        if not callable(orientation):
            orientation = get_orientation(orientation)

        return self._convert_cutout(
            convert_func=convert_pv,
            panel=panel,
            orientation=orientation,
            clearsky_model=clearsky_model,
            **params,
        )

    def pm25(self, **params):
        """
        Generate PM2.5 time series 	[ug / m3]
        (see convert_pm25 for details)

        Returns:
            xr.DataArray: PM2.5 time series

        """

        return self._convert_cutout(convert_func=convert_pm25, **params)


def ds_reformat_index(ds: xr.DataArray) -> xr.DataArray:
    """Format the dataArray generated from the convert function.

    Args:
        ds (xr.DataArray): dataArray generated from the convert function.

    Returns:
        xr.DataArray: DataArray with lat and lon as dimensions.
    """

    if "lat" in ds.dims and "lon" in ds.dims:
        return ds.sortby(["lat", "lon"])
    elif "lat" in ds.coords and "lon" in ds.coords:
        return (
            ds.reset_coords(["lon", "lat"], drop=True)
            .rename({"x": "lon", "y": "lat"})
            .sortby(["lat", "lon"])
        )
    return ds.rename({"x": "lon", "y": "lat"}).sortby(["lat", "lon"])


def _find_intercept(list1, list2, start, threshold=0):
    """Find_intercept is a helper function to find the best start point for doing coarsening
    in order to make the coordinates of the coarsen as close to the target as possible.
    """
    min_res = 0
    init = 0
    for i in range(len(list1) - start):
        resid = ((list1[start + i] - list2[0]) % (list2[1] - list2[0])).values.tolist()
        if i == 0:
            init = resid
        if resid <= threshold:
            return i
        if resid > min_res:
            min_res = resid
        else:
            min_res = resid
            break
    if min_res == init:
        return 0
    else:
        return i


def coarsen(ori: xr.Dataset, tar: xr.Dataset, func: Literal["sum", "mean"] = "mean"):
    """This function will reindex the original xarray dataset according to the coordiantes of the target.
    There might be a bias for lattitudes and longitudes. The bias are normally within 0.01 degrees.
    In order to not lose too much data, a threshold for bias in degree could be given.
    When threshold = 0, it means that the function is going to find the best place with smallest bias.

    Args:
        ori (xr.Dataset): The original xarray dataset.
        tar (xr.Dataset): The target xarray dataset.
        func (Literal['sum', 'mean']): The function to be used for reduction. Defaults to "mean".

    Returns:
        xr.Dataset: The reindexed xarray dataset.

    Raises:
        ValueError: reduction method can only be 'mean' or 'sum'.
    """
    lat_multiple = round(
        ((tar.lat[1] - tar.lat[0]) / (ori.lat[1] - ori.lat[0])).values.tolist()
    )
    lon_multiple = round(
        ((tar.lon[1] - tar.lon[0]) / (ori.lon[1] - ori.lon[0])).values.tolist()
    )
    lat_start = _find_intercept(ori.lat, tar.lat, (lat_multiple - 1) // 2)
    lon_start = _find_intercept(ori.lon, tar.lon, (lon_multiple - 1) // 2)

    if func == "mean":
        _coarsen = (
            ori.isel(lat=slice(lat_start, None), lon=slice(lon_start, None))
            .coarsen(
                dim={"lat": lat_multiple, "lon": lon_multiple},
                side={"lat": "left", "lon": "left"},
                boundary="pad",
            )
            .mean()
        )
    elif func == "sum":
        _coarsen = (
            ori.isel(lat=slice(lat_start, None), lon=slice(lon_start, None))
            .coarsen(
                dim={"lat": lat_multiple, "lon": lon_multiple},
                side={"lat": "left", "lon": "left"},
                boundary="pad",
            )
            .sum()
        )
    else:
        raise ValueError("func can only be 'mean' or 'sum'")

    return _coarsen.reindex_like(tar, method="nearest")


def calc_grid_area(lis_lats_lons):
    """Calculate area in km^2 for a grid cell given lats and lon border, with help from:
    https://stackoverflow.com/questions/4681737/how-to-calculate-the-area-of-a-polygon-on-the-earths-surface-using-python

    """
    lons, lats = zip(*lis_lats_lons)
    ll = list(set(lats))[::-1]
    var = []
    for i in range(len(ll)):
        var.append("lat_" + str(i + 1))
    st = ""
    for v, l in zip(var, ll):  # noqa: E741
        st = st + str(v) + "=" + str(l) + " " + "+"
    st = (
        st
        + "lat_0="
        + str(np.mean(ll))
        + " "
        + "+"
        + "lon_0"
        + "="
        + str(np.mean(lons))
    )
    tx = "+proj=aea +" + st
    pa = pyproj.Proj(tx)

    x, y = pa(lons, lats)
    cop = {"type": "Polygon", "coordinates": [zip(x, y)]}

    return shapely.geometry.shape(cop).area / 1000000


def calc_shp_area(shp, shp_projection="+proj=latlon"):
    """calculate area in km^2 of the shapes for each shp object"""
    temp_shape = shapely.ops.transform(
        partial(
            pyproj.transform,
            pyproj.Proj(shp_projection),
            pyproj.Proj(proj="aea", lat_1=shp.bounds[1], lat_2=shp.bounds[3]),
        ),
        shp,
    )
    return temp_shape.area / 1000000
