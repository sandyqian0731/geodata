Wind Modeling
=============

Starting from geodata v0.2.0, geodata's wind modeling capability lives in a separate
``geodata.model.wind`` module. The **supported ERA5 path** uses vertical spline
**interpolation** on ``wind_3d_hourly`` data (see :doc:`interpolation`).

How to use the models
---------------------

Unlike some other modules of geodata, the model module does not get imported
automatically. In other words, one cannot use the models with an import statement
like this:

.. code:: Python

    import geodata
    model = geodata.model.wind.WindInterpolationModel()

Instead, the user must import the models explicitly:

.. code:: Python

    from geodata.model.wind import WindInterpolationModel
    model = WindInterpolationModel()

The reason for this is to keep the main geodata namespace clean,
since we might add many more models in the future.

In general, models are created based off of a dataset object, we must be downloaded
first. We'll use the `wind_3d_hourly` dataset from the ERA5 dataset as an example:

.. code:: Python

    from geodata.datasets import load_dataset
    from geodata.model.wind import WindInterpolationModel

    # Load the dataset
    ds_cls = load_dataset("wind_3d_hourly")
    ds = ds_cls(
        years=slice(2006, 2006),
        months=slice(1, 1),
        bounds=[-10, 35, 10, 45]  # Optional: specify the bounding box
    )

    if not ds.downloaded:
        ds.download()  # Download the data if we don't have it locally

    print(ds.downloaded)  # Check if the dataset is downloaded. Should return True.


Once we have the dataset, we can create a model based on it.
The model will be associated with the dataset forever, so if you wish to use a
different dataset, you will need to create a new model.

.. code:: Python

    # Create a model based on the above dataset
    model = WindInterpolationModel(ds)
    model.prepare()  # Prepare the model

    print(model.prepared)  # Check if the model is prepared. Should return True.


Preparing the model (``prepare``, ``prepared``, ``force``)
----------------------------------------------------------

Wind models must be **prepared** before ``estimate()``. Preparation reads the
downloaded ERA5 files, computes month-by-month coefficients (B-spline parameters for
interpolation), and writes cached results under ``GEODATA_ROOT/models/`` (see
:doc:`/quick_start/packagesetup`).

- ``model.prepared`` — ``True`` when every month in the model's year/month range has
  cached outputs on disk.
- ``model.prepare()`` — run once after ``ds.downloaded`` is ``True``. Safe to skip if
  already prepared.
- ``model.prepare(force=True)`` — delete and recompute cached months (use after changing
  ``years`` / ``months`` / ``bounds`` on the source dataset, or when upgrading geodata).

``estimate()`` raises if the model is not prepared. Pvlib does **not** use this
prepare step; only wind models do.

.. code:: Python

    if not model.prepared:
        model.prepare()
    # After changing the source time range or domain:
    # model.prepare(force=True)

How caching works internally (``geodata.model.results``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``prepare()`` writes one cached file per day (or per month, depending on the
model's ``frequency``) under
``GEODATA_ROOT/models/<dataset module>/<ModelClassName>/<year>/<month>/``,
plus a ``meta.json`` recording a SHA-256 hash of every file written. This is
implemented by :code:`geodata.model.results` (``DailyModelResult`` /
``MonthlyModelResult``), which both ``model.prepared`` and ``model.prepare()``
delegate to.

The cache key is only **(dataset module, model class, year, month)** — it does
**not** include turbine, panel, or other ``estimate()`` arguments. This is
intentional: ``prepare()`` only computes the reusable interpolation
coefficients from the source weather data; turbine/panel-specific conversion
happens later in ``estimate()``. So switching ``turbine=`` between calls does
**not** require re-running ``prepare()`` — but changing the *source* dataset
(different years/months/bounds, or re-downloaded raw files) does, via
``force=True``.

By default (``quick_check=False``, the model constructor's default), checking
``model.prepared`` recomputes the SHA-256 hash of every cached file and
compares it against the hash recorded at prepare-time — this catches
silently truncated or manually edited cache files, but means the check itself
gets slower the more months/files you have cached (relevant once you're
running a decade of data across multiple countries). Pass
``quick_check=True`` to the model constructor to only check that the expected
files *exist*, skipping the hash comparison — much faster for repeated batch
resubmissions once you've already validated a run's integrity once, at the
cost of not detecting corrupted cache files.

.. code:: Python

    # Skip expensive hash verification on every `prepared` check —
    # useful once a run's cache has already been validated.
    model = WindInterpolationModel(ds, quick_check=True)

Pvlib does **not** go through any of this: :code:`Pvlib.prepare()` is a
no-op and :code:`Pvlib.prepared` always returns ``True`` (see
:doc:`/modeling/pvlib/index`). Every call to ``Pvlib.estimate()`` re-reads
the raw source files and recomputes from scratch — there is no cross-run
caching for the solar model.

Once the model is prepared, we can use it to estimate wind speed at desired heights.

.. code:: Python

    # Estimate wind speed at a specific height (e.g., 100 meters)
    wind_speed: xr.Dataset = model.estimate(height=100.0)

    # The wind_speed variable is an xarray Dataset containing the estimated wind speed at the specified height.
    # Over the region covered by the original dataset.


The above demonstrates the typical workflow. More model-specific details can be found
in each model's respective tutorial as well as in the API reference.

Estimate options
----------------

Wind models share the same ``estimate()`` subsetting interface (defined on
``BaseModel`` in ``geodata.model``).

Temporal subsetting
~~~~~~~~~~~~~~~~~~~

Use ``years`` and ``months`` slices to limit the estimation period. For example,
``years=slice(2006, 2006), months=slice(1, 1)`` processes January 2006 only.

Spatial subsetting (``xs``, ``ys``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``xs`` and ``ys`` as ``slice(start, stop)`` to restrict longitude (``x``) and
latitude (``y``). Omit either argument to use the full horizontal domain of the
prepared source.

Geodata **normalizes slice bounds** before ``xarray.Dataset.sel()``. You may pass
``slice(high, low)`` or ``slice(low, high)``; the helper resolves the inclusive
range and matches the coordinate's ascending or descending order (ERA5 latitude is
typically descending). Without this, a slice like ``ys=slice(46, 48)`` on a
descending ``y`` axis can incorrectly return an empty selection.

.. code:: Python

    # Subregion over central Europe — bounds order does not matter
    wind_speed = model.estimate(
        height=100.0,
        years=slice(2006, 2006),
        months=slice(1, 1),
        xs=slice(8, 10),
        ys=slice(48, 46),
    )

Wind-specific arguments
~~~~~~~~~~~~~~~~~~~~~~~

Pass **either**:

- ``height=<meters>`` — hub-height or AGL wind speed (``WindInterpolationModel`` on
  ``wind_3d_hourly``), or
- ``turbine="<name>"`` — capacity factor (``cf``) from a turbine YAML under
  ``geodata.resources.windturbine``. The name is the YAML stem (e.g.
  ``Vestas_V112_3MW``). See :doc:`interpolation` Step 5 for usage and what
  ``cf`` represents.

List available turbines with ``geodata.resource.get_available_windturbines()``.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials on Specific Models

   interpolation
