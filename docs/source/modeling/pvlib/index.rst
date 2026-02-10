PVLib Modeling
==============

PVLib is a Python library for modeling solar photovoltaic systems. It provides a set of tools for modeling the performance of solar photovoltaic systems

How to use the model
---------------------

The PVLib models are imported from the `pvlib` module.

Step 1: Import the necessary libraries
----------------------------------------

To get started, we need to import the required libraries. We will import
the `pvlib` from the `geodata` library, as well as any other
libraries needed for data handling and visualization.

.. code:: Python

    import xarray as xr

    from geodata.datasets import load_dataset
    from geodata.model.pvlib import Pvlib

Step 2: Load the dataset
------------------------

Next, we need to load the dataset that contains the solar irradiance data.
We will use the `wind_solar_hourly` dataset from the ERA5 dataset.

.. code:: Python

    # Load the dataset
    ds_cls = load_dataset("wind_solar_hourly")
    ds = ds_cls(
        years = slice(2016, 2016),
        months = slice(1, 1)
    )
    if not ds.downloaded:
        ds.download() # Download the data if we don't have it locally
    print(ds.downloaded) # Check if the dataset is downloaded. Should return True.

Step 3: Create the model with specific configs
----------------------------------------------

Next, we need to create the model with specific configs.

.. code:: Python

    model = Pvlib(ds)

Two configurations are required: (1) **PV system setup** — physical array geometry (tilt, azimuth), module and inverter from the SAM database, and racking; (2) **Model config** — algorithms for clearsky irradiance, transposition, solar position, airmass, DC/AC conversion (CEC, Sandia), and losses (AOI, spectral, ohmic).
Following is an example of how to create the model with specific configs.

.. code:: Python

    # create the pv_system
    n_mods = 50
    n_strings = 1
    cec_modules = model.retrieve_sam('CECMod')
    module = cec_modules['Kaneka_U_SA105']
    inv = model.retrieve_sam("CECInverter")['Fronius_USA__CL_33_3_Delta__208V_']
    model.init_pv_system(
        arrays = None,
        surface_tilt=35,
        surface_azimuth=180,
        racking_model = 'open_rack',
        module_parameters=module,
        modules_per_string = n_mods,
        module_type = 'glass_polymer',
        module = 'Kaneka_U_SA105',
        strings_per_inverter = n_strings, 
        inverter_parameters=inv
    )

.. code:: Python

    model.init_model_config(
        clearsky_model= 'haurwitz',
        transposition_model='perez', 
        solar_position_method= 'nrel_numpy',
        airmass_model= 'kastenyoung1989',
        dc_model='cec',
        ac_model='sandia', 
        aoi_model="physical",
        spectral_model='first_solar',
        dc_ohmic_model='no_loss'
    )

Step 4: Estimate the capacity factor
------------------------------------

Next, we can estimate the AC Power and PV capacity using the model.

.. code:: Python

    cf = model.estimate(
        years = slice(2016, 2016), 
        months = slice(1, 1),
        xs = slice(8, 10), # Optional: specify the bounding box
        ys = slice(48, 46), # here is an example bounding box for central europe
    )
    print(cf)

The output will be an xarray Dataset containing the estimated AC Power and PV capacity values for the specified region and time period.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials on Specific Models

