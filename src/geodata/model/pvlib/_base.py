# Copyright 2016-2017 Gorm Andresen (Aarhus University), Jonas Hoersch (FIAS), Tom Brown (FIAS)
# Copyright 2020 Michael Davidson (UCSD), William Honaker, Jiahe Feng (UCSD), Yuanbo Shi
# Copyright 2023-2024 Xiqiang Liu, 2025 Keyu Long

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
GEODATA

Geospatial Data Collection and "Pre-Analysis" Tools

TODO: Documentation here

"""
import pandas as pd
import xarray as xr
from pvlib import pvsystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from timezonefinder import TimezoneFinder

from .._base import BaseModel, _get_xr_engine, _should_use_parallel_reading
from geodata.logging import logger
from .calculations import calculate_pvlib_solarposition, calculate_ghi, calculate_relative_humidity, calculate_precipitable_water, convert_kelvin_to_celsius
from tqdm.auto import tqdm

class ModelChainConfig:
    """
    Defines pvlib ModelChain parameters as a class that
    can be passed to one or more instances of pvlib_model().
    Allows user to reuse a common set of ModelChain parameters across multiple
    PVSystems or even multiple cutouts.  

    Parameters
    ----------
    clearsky_model : string, default 'ineichen'
        Specifies the clear-sky model. Passed to location.get_clearsky. 
        Only used when DNI is not found in the weather inputs.
    transposition_model : string, default 'haydavies'
        Specifies the transposition model. Passed to system.get_irradiance.
    solar_position_method : string, default 'nrel_numpy'
        Specifies the method for calculating solar positions. Passed to location.get_solarposition.
    airmass_model : string, default 'kastenyoung1989'
        Specifies the airmass model. Passed to location.get_airmass.
    dc_model : string or function, optional
        Specifies the DC model. Valid strings are 'sapm', 'desoto', 'cec', 'pvsyst', 'pvwatts'. 
        If not specified, the model will be inferred from the parameters of system.arrays[i].module_parameters. 
        A user-defined function may also be provided, with the ModelChain instance passed as the first argument.
    ac_model : string or function, optional
        Specifies the AC model. Valid strings are 'sandia', 'adr', 'pvwatts'. 
        If not specified, the model will be inferred from the parameters of system.inverter_parameters. 
        A user-defined function may also be provided, with the ModelChain instance passed as the first argument.
    aoi_model : string or function, optional
        Specifies the angle of incidence (AOI) model. Valid strings are 'physical', 'ashrae', 'sapm', 'martin_ruiz', 
        'interp', 'no_loss'. If not specified, the model will be inferred from the parameters of 
        system.arrays[i].module_parameters. A user-defined function may also be provided, 
        with the ModelChain instance passed as the first argument.
    spectral_model : string or function, optional
        Specifies the spectral model. Valid strings are 'sapm', 'first_solar', 'no_loss'. 
        If not specified, the model will be inferred from the parameters of system.arrays[i].module_parameters. 
        A user-defined function may also be provided, with the ModelChain instance passed as the first argument.
    temperature_model : string or function, optional
        Specifies the temperature model. Valid strings are 'sapm', 'pvsyst', 'faiman', 'fuentes', 'noct_sam'. 
        A user-defined function may also be provided, with the ModelChain instance passed as the first argument.
    dc_ohmic_model : string or function, default 'no_loss'
        Specifies the DC ohmic loss model. Valid strings are 'dc_ohms_from_percent', 'no_loss'. 
        A user-defined function may also be provided, with the ModelChain instance passed as the first argument.
    losses_model : string or function, default 'no_loss'
        Specifies the losses model. Valid strings are 'pvwatts', 'no_loss'. 
        A user-defined function may also be provided, with the ModelChain instance passed as the first argument.
    name : string, optional
        Specifies the name of the ModelChain instance.
    
    For full documentation, see:
    - pvlib.modelchain.ModelChain(): 
        https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.modelchain.ModelChain.html

    """
    def __init__(
        self,
        clearsky_model='ineichen',
        transposition_model='haydavies',
        solar_position_method='nrel_numpy',
        airmass_model='kastenyoung1989',
        dc_model=None,
        ac_model=None,
        aoi_model=None,
        spectral_model=None,
        temperature_model=None,
        dc_ohmic_model='no_loss',
        losses_model='no_loss',
        name=None
    ):
        self.clearsky_model = clearsky_model
        self.transposition_model = transposition_model
        self.solar_position_method = solar_position_method
        self.airmass_model = airmass_model
        self.dc_model = dc_model
        self.ac_model = ac_model
        self.aoi_model = aoi_model
        self.spectral_model = spectral_model
        self.temperature_model = temperature_model
        self.dc_ohmic_model = dc_ohmic_model
        self.losses_model = losses_model
        self.name = name

    def model_chain_to_kwargs(self):
        return self.__dict__
    
class Pvlib(BaseModel):
    """The pvlib model"""

    type: str = "pvlib"

    SUPPORTED_WEATHER_DATA_CONFIGS = ("wind_solar_hourly",)

    @property
    def prepared(self) -> bool:
        """This model does not need to be prepared"""
        return True
    
    def prepare(self, force: bool = False):
        """Skip preparation - this model doesn't need it."""
        logger.info("This model does not require preparation. Skipping.")
        return
    
    def init_model_config(
            self,
            clearsky_model='ineichen',
            transposition_model='haydavies',
            solar_position_method='nrel_numpy',
            airmass_model='kastenyoung1989',
            dc_model=None,
            ac_model=None,
            aoi_model=None,
            spectral_model=None,
            temperature_model=None,
            dc_ohmic_model='no_loss',
            losses_model='no_loss',
            name=None
        ):
            self.config = ModelChainConfig(
                clearsky_model= clearsky_model,
                transposition_model= transposition_model, 
                solar_position_method= solar_position_method,
                airmass_model= airmass_model,
                dc_model= dc_model,
                ac_model= ac_model, 
                aoi_model= aoi_model,
                spectral_model= spectral_model,
                temperature_model= temperature_model,
                dc_ohmic_model= dc_ohmic_model,
                losses_model= losses_model,
                name= name
            )

    def retrieve_sam(self, samfile, path=None):
        """
        Wrapper for pvlib.pvsystem.retrieve_sam(). Retrieves latest module 
        and inverter info from a file bundled with pvlib, a path or a 
        URL (like SAM’s website), and returns it as a Pandas DataFrame.

        Supported databases:
        - CEC module database
        - Sandia Module database
        - CEC Inverter database
        - Anton Driesse Inverter database

        Parameters
        ----------
        name : string
                Use one of the following strings to retrieve a database bundled with pvlib:
                    - ’CECMod’ - returns the CEC module database
                    - ’CECInverter’ - returns the CEC Inverter database
                    - ’SandiaInverter’ - returns the CEC Inverter database 
                        (CEC is only current inverter db available; tag kept for backwards compatibility)
                    - ’SandiaMod’ - returns the Sandia Module database
                    - ’ADRInverter’ - returns the ADR Inverter database
        
        Optional Parameters
        ----------
        path : string
                Path to a CSV file or a URL.

        Returns: DataFrame

        See also:
            - pvlib.pvsystem.retrieve_sam(): 
                https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.retrieve_sam.html

        """
        return pvsystem.retrieve_sam(name=samfile, path=path)
    
    def init_pv_system(self, *args, **kwargs):
        """
        Wrapper for pvlib.pvsystem.PVSystem().
        The PVSystem class defines a standard set of PV system attributes
        and modeling functions. This class describes the collection and 
        interactions of PV system components rather than an installed system
        on the ground. It is typically used in combination with Location 
        and ModelChain objects.

        The class supports basic system topologies consisting of:
            - N total modules arranged in series (modules_per_string=N, strings_per_inverter=1).
            - M total modules arranged in parallel (modules_per_string=1, strings_per_inverter=M).
            - NxM total modules arranged in M strings of N modules each 
            (modules_per_string=N, strings_per_inverter=M).

        For full documentation, see: https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.PVSystem.html

        Parameters
        ----------
        arrays : array (optional)
                An Array or list of arrays that are part of the system. 
                See pvlib documentation for full description.
        surface_tilt : float
                Surface tilt angles in decimal degrees. The tilt angle is 
                defined as degrees from horizontal (e.g. surface facing up = 0, 
                surface facing horizon = 90).
        surface_azimuth : float
                Azimuth angle of the module surface. North=0, East=90, South=180, West=270.
        albedo : float
                Ground surface albedo. If not supplied, then surface_type is used to look up 
                a value in pvlib.albedo.SURFACE_ALBEDOS. If surface_type is also not supplied 
                then a ground surface albedo of 0.25 is used.
        surface_type : string
                The ground surface type. See pvlib.albedo.SURFACE_ALBEDOS for valid values.
        module : string
                The model name of the modules. May be used to look up the module_parameters dictionary via some other method.
        module_type : string 
                Describes the module’s construction. Valid strings are ‘glass_polymer’ and ‘glass_glass’. 
                Used for cell and module temperature calculations.
        module_parameters : dict
                Module parameters as defined by the SAPM, CEC, or other.
        temperature_model_parameters : dict
                Temperature model parameters as required by one of the models in pvlib.temperature (excluding poa_global, temp_air and wind_speed).
        modules_per_string : int, float
                See system topology discussion above.
        strings_per_inverter : int, float 
                See system topology discussion above.
        inverter : string 
                The model name of the inverters. May be used to look up the inverter_parameters dictionary via some other method.
        inverter_parameters : dict
                Inverter parameters as defined by the SAPM, CEC, or other.
        racking_model : string 
                Valid strings are ‘open_rack’, ‘close_mount’, and ‘insulated_back’. 
                Used to identify a parameter set for the SAPM cell temperature model.
        losses_parameters : dict 
                Losses parameters as defined by PVWatts or other.    
        name : string (optional)
        
        """
        self.pv_system = pvsystem.PVSystem(*args, **kwargs)


    def _estimate_dataset(self, params: xr.Dataset, **kwargs) -> xr.Dataset | xr.DataArray:  # type: ignore[override]
        """Estimate PV output from prepared dataset.
        
        Args:
            params: Dataset (already filtered by years/months/xs/ys from BaseModel)
            **kwargs: Additional parameters (not used currently, but available)
        
        Returns:
            Dataset with AC power and PV capacity (returns Dataset, but BaseModel expects DataArray)
        """

        result = self._pvlib_model(params, self.pv_system, self.config)
        return result
    
    def estimate(self,
        years: slice | None = None,
        months: slice | None = None,
        xs: slice | None = None,
        ys: slice | None = None,
        **kwargs,
        ) -> xr.DataArray:
            """Get pvlib model results.
            
            This method processes data month-by-month to avoid memory issues with large datasets.
            Results from each month are concatenated along the time dimension.
            
            Args:
                years: Year range (slice)
                months: Month range (slice)
                xs: X-coordinate range (slice)
                ys: Y-coordinate range (slice)
                **kwargs: Additional parameters
            
            Returns:
                Dataset with AC power and PV capacity, concatenated across all months
            """
            if getattr(self, 'pv_system', None) is None:
                raise ValueError("pv_system is not initialized. Call init_pv_system() first.")
            if getattr(self, 'config', None) is None:
                raise ValueError("model_config is not initialized. Call init_model_config() first.")
            
            # Get result objects for the requested time range
            if years is None and months is None:
                results = self.flattened_results
            elif months is None:
                # If years specified but months not, use all months
                if years is None:
                    results = self.flattened_results
                else:
                    results = self.get_result_year_month(years, slice(1, 13))
            else:
                # Both years and months specified
                if years is None:
                    # If only months specified, need to get all years
                    # Use the source dataset's year range
                    years = self.source.years
                results = self.get_result_year_month(years, months)
            
            if not results:
                raise ValueError("No results found for the specified year/month range.")
            
            # Process month-by-month to manage memory
            logger.info(
                f"Processing {len(results)} month(s) month-by-month to manage memory usage"
            )
            
            engine = _get_xr_engine()
            parallel = _should_use_parallel_reading()
            
            monthly_results = []
            
            for result in tqdm(results, desc="Processing months", unit="month"):
                # Load only this month's raw data files
                ref_files = result.ref_files
                
                if not ref_files:
                    logger.warning(
                        f"No files found for {result.year:04d}-{result.month:02d}, skipping."
                    )
                    continue
                
                logger.debug(
                    f"Loading {len(ref_files)} file(s) for {result.year:04d}-{result.month:02d} "
                    f"with engine={engine}, parallel={parallel}"
                )
                
                # Open this month's dataset
                with xr.open_mfdataset(
                    ref_files,
                    engine=engine,
                    parallel=parallel,
                ) as params:
                    # Apply spatial filtering if specified
                    if xs is not None:
                        params = params.sel(x=xs)
                    if ys is not None:
                        params = params.sel(y=ys)
                    
                    # Transform raw dataset to standardized format
                    # This applies the same transformations as prepare_func
                    # (renames variables, calculates derived quantities, etc.)
                    dataset_cls = type(self.source)
                    if hasattr(dataset_cls, 'transform_wind_solar_dataset'):
                        # Call the classmethod to transform the dataset
                        params = dataset_cls.transform_wind_solar_dataset(params)  # type: ignore[attr-defined]
                    else:
                        logger.warning(
                            "Dataset does not have transform_wind_solar_dataset method. "
                            "Assuming data is already in the correct format."
                        )
                    
                    # Process this month's data
                    monthly_output = self._estimate_dataset(params, **kwargs)
                    
                    # Store the result (will concatenate later)
                    monthly_results.append(monthly_output)
            
            if not monthly_results:
                raise ValueError("No data was successfully processed for the specified range.")
            
            # Concatenate all monthly results along the time dimension
            logger.info(f"Concatenating {len(monthly_results)} month(s) of results")
            
            # Ensure all datasets have compatible coordinates
            # Sort by time to ensure proper ordering
            combined_result = xr.concat(monthly_results, dim='time')
            
            # Sort by time to ensure chronological order
            if 'time' in combined_result.coords:
                combined_result = combined_result.sortby('time')
            
            return combined_result
        
    def _prepare_pvlib_ds(self, ds: xr.Dataset, *varnames: str) -> xr.Dataset:
        """
        Prepares an `xarray.Dataset` from a geodata `cutout` class for use in model simulations using `pvlib`.
        This function extracts specified variables from the `cutout` dataset, calculates additional parameters 
        like global horizontal irradiance (GHI), precipitable water, and solar position, and renames fields to 
        align with expected inputs.

        Requires a cutout with the following variables:

        - **influx_diffuse** (*float*) - Diffuse horizontal irradiance.  
        - **influx_direct** (*float*) - Direct normal irradiance.  
        - **dewpoint_temperature** (*float*) - Dewpoint temperature in Celsius.  
        - **temperature** (*float*) - Air temperature in Celsius.  
        - **wnd100m** (*float*) - Wind speed at 100m.  

        Outputs an `xarray.Dataset` with the following variables:

        - **dhi** (*float*) - Diffuse horizontal irradiance.  
        - **dni** (*float*) - Direct normal irradiance.  
        - **ghi** (*float*) - Global horizontal irradiance (calculated via :code:`_calculate_ghi()`).  
        - **temp_air** (*float*) - Air temperature in Celsius.  
        - **wind_speed** (*float*) - Wind speed at 100m.  
        - **precipitable_water** (*float*) - Precipitable water (calculated via :code:`_calculate_precipitable_water()`).

        Parameters
        ----------
        ds : xarray.Dataset
            Must contain following variables: influx_diffuse, influx_direct,
            dewpoint_temperature, temperature, wnd100m.
        varnames : string
            String values representing names of required variables.

        Returns
        -------
        weather_data : `xarray.Dataset`
            Dataset containing necessary variables to run `pvlib` model simulations.

        """

        if varnames:
            # Check which variables are actually available
            available_vars = [v for v in varnames if v in ds.data_vars]
            missing_vars = [v for v in varnames if v not in ds.data_vars]
            if missing_vars:
                logger.warning(f"Missing variables: {missing_vars}. Available: {list(ds.data_vars.keys())}")
            if available_vars:
                ds = ds[available_vars]
            else:
                logger.error(f"None of the requested variables {varnames} are available in dataset")
                raise KeyError(f"None of the requested variables {varnames} are available. Available variables: {list(ds.data_vars.keys())}")

        temperature_celsius = convert_kelvin_to_celsius(ds.temperature)

        relative_humidity = calculate_relative_humidity(
            temperature_celsius,
            #_convert_celsius(ds.dewpoint_temperature),
            convert_kelvin_to_celsius(ds.d2m),
        )

        precipitable_water = calculate_precipitable_water(
            temperature_celsius,
            relative_humidity
        )

        sp = calculate_pvlib_solarposition(ds)
        ghi = calculate_ghi(ds, sp['zenith'])

        ds = (
            ds
            .assign(
                ghi=ghi,
                temperature=temperature_celsius,
                precipitable_water=precipitable_water
            )
            .rename({
                'influx_diffuse': 'dhi',
                'influx_direct': 'dni',
                'temperature': 'temp_air',
                'wnd100m': 'wind_speed'
            })
        )

        return ds[[
            "dhi", 
            "dni",
            "ghi", 
            "temp_air", 
            "wind_speed", 
            "precipitable_water"
        ]]
    
    def _pvlib_model(
        self,
        ds: xr.Dataset, 
        system: pvsystem.PVSystem,
        model_chain_config: ModelChainConfig,
        vars: list[str] = ["influx_diffuse", "influx_direct", "dewpoint_temperature", "temperature", "wnd100m"]
    ) -> xr.Dataset:
        
        """
        Applies a `pvlib` model using :code:`pvlib.modelchain.ModelChain()` across all unique coordinates 
        represented in a `geodata` cutout. This function prepares input weather data, initializes the 
        `pvlib` model, and runs simulations for each set of coordinates, outputting an xarray dataset 
        containing all simulation results.

        Requires a cutout with the following variables:

        - **influx_diffuse** (*float*) - Diffuse horizontal irradiance.  
        - **influx_direct** (*float*) - Direct normal irradiance.  
        - **dewpoint_temperature** (*float*) - Dewpoint temperature in Celsius.  
        - **temperature** (*float*) - Air temperature in Celsius.  
        - **wnd100m** (*float*) - Wind speed at 100m.  

        Outputs an `xarray.Dataset` containing:

        - **ac** (*float*) - AC photovoltaic output (W).
        - **pv** (*float*) - Photovoltaic capacity.

        Parameters
        ----------
        cutout : geodata **cutout** class
            Cutout generated by the `geodata` library, based on the ERA5 dataset.  
            Must contain the required meteorological variables.
        system : pvlib **PVSystem** class
            The photovoltaic system to be simulated.  Generated by :code:`geodata.pvlib.pv_system()`
        model_chain_config : `ModelChainConfig`
            Configuration object for :code:`pvlib.modelchain.ModelChain()` with model parameters.
        vars : list of str, optional
            List of variable names required for simulation. Defaults to:
            ['influx_diffuse', 'influx_direct', 'dewpoint_temperature', 'temperature', 'wnd100m'].

        Returns
        -------
        xr.Dataset
            Dataset containing ac power output and pv capacity across all coordinates in the cutout.

        """
        ptc = system.arrays[0].module_parameters['PTC']
        n_mods = system.arrays[0].modules_per_string

        weather_data = self._prepare_pvlib_ds(ds, *vars).to_dataframe()
        unique_coords = weather_data.index.droplevel('time').drop_duplicates()
        coord_subsets = []
        for y, x in unique_coords:
            subset = weather_data.loc[(slice(None), y, x), :].reset_index(['x', 'y'])
            tz_str = TimezoneFinder().timezone_at(lat=y, lng=x)
            if tz_str is None:
                raise ValueError(f"Timezone not found for coordinates ({y}, {x})")
            location = Location(latitude=y, longitude=x, tz = tz_str) # type: ignore[arg-type]
            
            mc = ModelChain(
                system, 
                location, 
                **model_chain_config.model_chain_to_kwargs()
            )
            mc.run_model(subset)
            
            subset['ac'] = mc.results.ac
            subset.loc[subset['ac'] < 0, 'ac'] = 0
            subset['pv'] = subset['ac'] / (ptc * n_mods)

            coord_subsets.append(subset)

        weather_data_final = pd.concat(coord_subsets)

        return xr.Dataset.from_dataframe(weather_data_final)
    
    def _prepare_dataset(self, source: xr.Dataset) -> xr.Dataset:
        """This will never be called, but must be implemented (abstract method)."""
        raise NotImplementedError("This model does not use _prepare_dataset")