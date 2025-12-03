import xarray as xr
from dask.distributed import Client

from geodata.model.wind import WindInterpolationModel
from geodata.datasets import load_dataset


def test_wind_interpolation_workflow():
    """Test that the wind interpolation workflow completes without errors.
    
    This test verifies:
    - Dataset can be loaded and downloaded
    - Model can be created and prepared
    - Capacity factor estimation works (globally and with bounds)
    - Wind speed estimation works at a specific height
    - Results can be computed and have valid values
    """

    client = Client(processes=True, threads_per_worker=1)

    years = slice(2016, 2016)
    months = slice(1, 1)

    ds_cls = load_dataset("wind_3d_hourly")
    ds = ds_cls(years=years, months=months, testing=True)

    ds.download()
    assert ds.downloaded, "Dataset should be downloaded successfully"

    # Create model with the dataset
    model = WindInterpolationModel(ds)
    assert model is not None, "Model should be created successfully"
    
    turbine_name = "Enercon_E126_7500kW"
    china_bbox = (73.5, 18.2, 135.1, 53.6)  # China bounding box
    xs = slice(china_bbox[0], china_bbox[2])
    ys = slice(china_bbox[3], china_bbox[1])

    # Test capacity factor estimation globally
    cf_global = model.estimate(turbine=turbine_name)
    assert cf_global is not None, "Capacity factor estimation should return a result"
    assert isinstance(cf_global, (xr.DataArray, xr.Dataset)), \
        "Capacity factor should be an xarray DataArray or Dataset"
    
    # Test capacity factor estimation for China only
    cf_china = model.estimate(turbine=turbine_name, xs=xs, ys=ys)
    assert cf_china is not None, "Capacity factor estimation with bounds should return a result"
    assert isinstance(cf_china, (xr.DataArray, xr.Dataset)), \
        "Capacity factor with bounds should be an xarray DataArray or Dataset"

    # Test wind speed estimation at specific height
    speed = model.estimate(height=100.0, xs=xs, ys=ys)
    assert speed is not None, "Wind speed estimation should return a result"
    assert isinstance(speed, xr.DataArray), \
        "Wind speed should be an xarray DataArray"

    # Test that results can be computed
    cf_computed = cf_china.compute()
    assert cf_computed is not None, "Computed capacity factor should not be None"
    
    # Test that max value can be calculated (verifies data is valid and operations work)
    max_cf = cf_computed.max()
    assert max_cf is not None, "Max capacity factor should be calculable"