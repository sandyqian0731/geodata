# ERA5 model outputs

ERA5 reanalysis data is consumed through the modern ``load_dataset`` workflow
(see :ref:`downloading-era5-data` in [Dataset module overview](overview.rst)), then
converted to analysis-ready variables with the **modeling** modules below.

This page lists common outputs for the **current tested path**. It does not describe
legacy ``Cutout`` / ``geodata.convert`` outputs.

## Wind generation time-series

Hub-height **capacity factor** (``cf``) or **wind speed** at a chosen height from
ERA5 3D wind data:

- Dataset: ``wind_3d_hourly`` (or ``wind_3d_hourly_test`` for offline fixtures)
- Model: ``WindInterpolationModel`` — see [Wind modeling](../modeling/wind/index.rst)
  and [interpolation tutorial](../modeling/wind/interpolation.rst)

```python
from geodata.datasets import load_dataset
from geodata.model.wind import WindInterpolationModel

ds = load_dataset("wind_3d_hourly")(years=slice(2016, 2016), months=slice(1, 1))
if not ds.downloaded:
    ds.download()

model = WindInterpolationModel(ds)
model.prepare()
cf = model.estimate(turbine="Vestas_V112_3MW", years=slice(2016, 2016), months=slice(1, 1))
```

## Wind speed time-series

Same setup as above; pass ``height=<meters>`` instead of ``turbine=``:

```python
wind_speed = model.estimate(height=100.0, years=slice(2016, 2016), months=slice(1, 1))
```

## Solar photovoltaic generation time-series

Hourly **AC power** (``ac``) and **capacity factor** (``pv``) from ERA5 single-level
radiation and wind fields:

- Dataset: ``wind_solar_hourly`` (or ``wind_solar_hourly_test`` for offline fixtures)
- Model: ``Pvlib`` — see [PVLib modeling](../modeling/pvlib/index.rst)

After ``init_pv_system()`` and ``init_model_config()``, call ``estimate()`` (see the
pvlib docs for ``compact_output`` and spatial subsetting).

## See also

- [ERA5 CDS setup](era5.rst)
- [Offline ERA5 fixtures](../development/offline-era5-fixture-datasets.md)
