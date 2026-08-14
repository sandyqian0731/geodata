==========================
Dataset Module Overview
==========================

The ``geodata.datasets`` module provides tools and classes for accessing, managing, and
processing geospatial datasets. It offers a unified interface for loading various
data formats, handling metadata, and performing common geospatial operations.

Key Features
------------

- Supports the download and management of **ERA5** datasets via ``load_dataset`` (see :doc:`era5` for CDS account setup).

- **MERRA2** remains in the codebase in two forms: the original ``Dataset``/``Cutout``-based
  path, documented under :doc:`/legacy/index`; and a newer, registered
  ``geodata.datasets.merra2`` submodule (split into ``daily``/``hourly``/``monthly``,
  mirroring the ``era5`` restructure) that has **no documentation of its own** — the
  legacy MERRA2 tutorials describe the old API, not this one. Like HRRR above, this
  project does not use MERRA2 for any of the four target countries; ERA5 is the sole
  weather source for consistency across China, USA, Indonesia, and Vietnam.

- **HRRR** (``geodata.datasets.hrrr``, via `Herbie <https://herbie.readthedocs.io/>`_) is
  present in the codebase as a higher-resolution, US-only alternative to ERA5, but it has
  **no test coverage and no further documentation beyond this note** — treat it as
  experimental. It is not used anywhere in this project's pipeline: all four target
  countries (China, USA, Indonesia, Vietnam) are generated from ERA5 for a consistent
  data source and methodology across countries, including for the USA.

Typical Usage
-------------

Registered datasets are loaded by name, instantiated with a time range (and optional
geographic bounds), then downloaded with ``download()`` if the files are not already
on disk. The sections below use **ERA5** as the primary example; the same pattern
applies to other configs returned by ``list_datasets()``.

.. _downloading-era5-data:

Downloading ERA5 data
---------------------

Geodata's ERA5 dataset classes wrap the `Copernicus CDS API <https://cds.climate.copernicus.eu/>`_.
You configure credentials once (see :doc:`era5`), then download through Python — you do
**not** need to call ``cdsapi`` directly for normal use.

Files are stored under ``GEODATA_ROOT / era5 / <weather_config> / …`` (see
:doc:`../quick_start/packagesetup` for ``GEODATA_ROOT``).

**Example — 3D wind (for :doc:`../modeling/wind/index`):**

.. code-block:: python

    from geodata.datasets import load_dataset

    ds_cls = load_dataset("wind_3d_hourly")
    ds = ds_cls(
        years=slice(2016, 2016),
        months=slice(1, 1),
        bounds=[-10, 35, 10, 45],  # lon_min, lat_min, lon_max, lat_max
    )

    print(ds.downloaded)  # False until files exist locally

    if not ds.downloaded:
        ds.download()

    print(ds.downloaded)  # True when the catalog is complete

**Example — 2D wind and solar hourly (for :doc:`../modeling/pvlib/index`):**

.. code-block:: python

    from geodata.datasets import load_dataset

    ds_cls = load_dataset("wind_solar_hourly")
    ds = ds_cls(years=slice(2016, 2016), months=slice(1, 1))

    if not ds.downloaded:
        ds.download()

``bounds`` is optional; omit it to use the full spatial extent allowed by the dataset
class. With ``testing=True``, only a **small subset** of the catalog is requested (useful
for trying a download before committing to a full month):

.. code-block:: python

    ds = ds_cls(
        years=slice(2016, 2016),
        months=slice(1, 1),
        bounds=[50, 0, 48, 3],
        testing=True,
    )
    ds.download()

After ``downloaded`` is ``True``, pass ``ds`` to a model (for example
``WindInterpolationModel(ds)`` or ``Pvlib(ds)``).

Offline / CI without CDS
~~~~~~~~~~~~~~~~~~~~~~~~

For tests and local development without calling the CDS, use the committed fixture
configs ``wind_3d_hourly_test`` and ``wind_solar_hourly_test`` — same API, no
``download()`` required when fixture files are present. See
:doc:`../development/offline-era5-fixture-datasets`.

Dataset Classes
-----------------

The ``geodata.datasets`` module includes several dataset classes, each tailored for
specific datasets. These classes encapsulate the logic for downloading, processing, and
accessing the data. Some of the available ERA5 configs
(listed by ``weather_config``) include:

- ``wind_solar_hourly``: Hourly wind (10 m and 100 m) and solar radiation from ERA5
  single levels. Also referred to as the 2D wind and solar dataset.

- ``wind_3d_hourly``: Hourly wind on ERA5 model levels (131–137), stored as **daily**
  NetCDF files. Used by the wind interpolation model for hub-height wind speed.

You can use ``list_datasets()`` to see all registered names:

.. code-block:: python

    from geodata.datasets import list_datasets

    print(list_datasets())

Check whether data is on disk
------------------------------

The ``downloaded`` property is ``True`` when every file in the dataset **catalog** exists
under ``storage_root``. If any file is missing, call ``download()`` (or ``download(force=True)``
to re-fetch):

.. code-block:: python

    if not ds.downloaded:
        ds.download()

Dataset's Interoperability with Cutout
------------------------------------------------

At the moment, the dataset classes are not interoperable with the ``Cutout`` class.
In the future, we plan to consolidate the functionalities of the ``Cutout`` class into the
dataset classes and the modeling module (see :doc:`../modeling/wind/index`).

For now, after downloading a dataset, pass it to a modeling class — see
:doc:`../modeling/wind/index` or :doc:`../modeling/pvlib/index`.
