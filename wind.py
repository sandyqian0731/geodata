from dask.distributed import Client

import os
import xarray as xr

# 1. Define the required path
GEODATA_PATH = '/tscc/projects/ps-davidson/geodata'

# 2. Check if the path exists
if os.path.exists(GEODATA_PATH):
    # 3. If it exists, set the environment variable
    os.environ['GEODATA_ROOT'] = GEODATA_PATH
    print(f"✅ Successfully set GEODATA_ROOT to: {GEODATA_PATH}")
else:
    # 4. If it doesn't exist, print a warning or informational message
    print(f"⚠️ WARNING: Required path does not exist. Skipping setting GEODATA_ROOT: {GEODATA_PATH}")

from geodata.model.wind import WindInterpolationModel
from geodata.datasets import load_dataset

def main():
    # only for TSCC
    client = Client(processes=True, threads_per_worker=1)

    years = slice(2016, 2016)
    months = slice(1, 1)

    ds_cls = load_dataset("wind_3d_hourly")
    ds = ds_cls(years=years, months=months)
    ds.download()

    model = WindInterpolationModel(ds)
    print(model)

    turbine_name = "Enercon_E126_7500kW"
    china_bbox = (73.5, 18.2, 135.1, 53.6) # China bounding box
    xs = slice(china_bbox[0], china_bbox[2])
    ys = slice(china_bbox[3], china_bbox[1])

    cf = model.estimate(turbine=turbine_name) # Computes the capacity factor globally
    cf = model.estimate(turbine=turbine_name, xs=xs, ys=ys) # Computes the capacity factor for China only

    speed = model.estimate(height=100., xs=xs, ys=ys) # Computes the wind speed at 100m height for China only

    cf = cf.compute()
    cf.max()

    client.close()

if __name__ == "__main__":
    main()