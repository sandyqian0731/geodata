# Copyright 2023-2025 Michael Davidson (UCSD), Xiqiang Liu (UCSD)

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


import abc
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Self

import xarray as xr
from tqdm.auto import tqdm

from ..config import model_dir
from ..datasets._base import BaseDataset
from ..logging import logger
from ..utils import check_hash


@dataclass
class ModelResult:
    """Model result class. This class is used to store the result of a model.
    It contains the year, month, reference path, model path, and the hashes of the
    reference and model datasets.

    Args:
        year (int): Year of the model.
        month (int): Month of the model.
        ref_path (Path): Path to the reference dataset.
        path (Path): Path to the model dataset.
        ref_hash (str): Hash of the reference dataset.
        path_hash (str): Hash of the model dataset.
    """

    year: int
    month: int
    model: "BaseModel"

    _hashes: dict[str, str] = field(default_factory=dict)
    _prepared: bool = False

    @property
    def frequency(self) -> str:
        """Frequency of the model."""
        return self.model.frequency

    @property
    def module(self) -> str:
        """Module of the model."""
        return self.model.source.module

    @property
    def ref_path(self) -> Path:
        """Path to the reference dataset."""
        return (
            self.model._ref_path
            / self.model.source.weather_config
            / f"{self.year:04d}"
            / f"{self.month:02d}"
        )

    @property
    def files(self) -> list[Path]:
        """List of files in the model dataset. This could potentially include any
        files that are not prepared yet."""

        atomic_files = self.model.source.get_monthly_catalog(self.year, self.month)
        if atomic_files is None:
            raise ValueError(
                f"Model files for {self.year:04d}-{self.month:02d} not found."
            )

        files = []
        for f in atomic_files:
            p = f.path.relative_to(self.ref_path).with_stem(f.path.stem + ".params")
            files.append(self.path / p)
        return files

    @property
    def ref_files(self) -> list[Path]:
        """List of files in the reference dataset."""

        atomic_files = self.model.source.get_monthly_catalog(self.year, self.month)
        if atomic_files is None:
            raise ValueError(
                f"Reference files for {self.year:04d}-{self.month:02d} not found."
            )

        return [f.path for f in atomic_files if f.path.exists()]

    @property
    def ref_params(self) -> dict[Path, Path]:
        """Reference parameters of the model."""

        ref_params = {}
        for file in self.ref_files:
            if file.name.endswith(".params.nc"):
                ref_params[file] = self.path / file.name
            else:
                ref_params[file] = self.path / f"{file.stem}.params.nc"
        return ref_params

    @property
    def path(self) -> Path:
        """Path to the model dataset."""
        return (
            model_dir
            / self.module
            / self.model.__class__.__name__
            / f"{self.year:04d}"
            / f"{self.month:02d}"
        )

    @property
    def prepared(self) -> bool:
        """Check if the model is prepared.

        Returns:
            bool: True if prepared.
        """
        if not self._prepared:
            self._prepared = self._check_prepared()
        return self._prepared

    def _check_prepared(self) -> bool:
        """Check if the model is prepared.

        Returns:
            bool: True if prepared.
        """

        assert self.path is not None, "The model saving path has not been set yet."

        if not (self.path / "meta.json").exists():
            logger.warning(
                "Model %s-%s does not have metadata. Please prepare the model first.",
                self.year,
                self.month,
            )
            return False

        match self.frequency:
            case "daily":
                with ThreadPoolExecutor(
                    max_workers=os.getenv("MAX_WORKERS")
                ) as executor:
                    files = [(f, self._hashes.get(f.name)) for f in self.files]
                    results = list(
                        tqdm(
                            executor.map(lambda t: check_hash(*t), files),
                            total=len(files),
                            unit="file",
                            dynamic_ncols=True,
                            desc=f"Model Files Integrity Check {self.year:04d}-{self.month:02d}",
                        )
                    )
                for file, (is_valid, hash_value) in zip(files, results):
                    if not is_valid:
                        logger.warning(
                            "File %s in model has been modified since model creation. Model is not prepared!",
                            file,
                        )
                        return False
                return True
            case "monthly":
                return check_hash(self.path / f"{self.month:02d}.params.nc")[0]
            case _:
                raise ValueError(
                    f"Frequency {self.frequency} is not supported. Supported frequencies are: daily, monthly."
                )

    def register(self, dataset: xr.Dataset):
        """Register the model result with the dataset.

        Args:
            dataset (xr.Dataset): Dataset to register.
        """
        if not isinstance(dataset, xr.Dataset):
            raise ValueError(
                f"Dataset must be an xarray Dataset, but got {type(dataset)}."
            )

        match self.frequency:
            case "daily":
                day = dataset.get("valid_time").dt.day.values[0]
                dataset.to_netcdf(self.path / f"{day:02d}.params.nc")
                with open(self.path / f"{day:02d}.params.nc", "rb") as f:
                    self._hashes[f"{day:02d}.params.nc"] = hashlib.sha256(
                        f.read()
                    ).hexdigest()

            case "monthly":
                dataset.to_netcdf(self.path / f"{self.month:02d}.params.nc")
                with open(self.path / f"{self.month:02d}.params.nc", "rb") as f:
                    self._hashes[f"{self.month:02d}.params.nc"] = hashlib.sha256(
                        f.read()
                    ).hexdigest()

            case _:
                raise ValueError(
                    f"Frequency {self.frequency} is not supported. Supported frequencies are: daily, monthly."
                )

    @classmethod
    def from_year_month(cls, model: "BaseModel", year: int, month: int) -> Self:
        """Create an AtomicModel from year and month.

        Args:
            model (BaseModel): BaseModel object.
            year (int): Year of the model.
            month (int): Month of the model.

        Returns:
            AtomicModel: AtomicModel object.
        """

        if not (1 <= month <= 12):
            raise ValueError(f"Month {month} is not valid. Must be between 1 and 12.")
        if not (2000 <= year <= 2100):
            raise ValueError(
                f"Year {year} is not valid. Must be between 2000 and 2100."
            )

        path = (
            model_dir
            / model.source.module
            / model.__class__.__name__
            / f"{year:04d}"
            / f"{month:02d}"
            / "meta.json"
        )

        if not path.exists():
            return cls(model=model, year=year, month=month)

        with open(path, "r") as f:
            data = json.load(f)

        if data["year"] != year or data["month"] != month:
            raise ValueError(
                f"Model year {data['year']} and month {data['month']} do not match {year} and {month}."
            )

        return cls.from_dict(data, model)

    def __repr__(self):
        return f"AtomicModel(year={self.year}, month={self.month}, ref_path={self.ref_path}, path={self.path} {len(self.files)} / {len(self.ref_files)})"

    @classmethod
    def from_dict(cls, data: dict, model: "BaseModel") -> Self:
        """Create an AtomicModel from a dictionary.

        Args:
            data (dict): Dictionary with the model data.

        Returns:
            AtomicModel: AtomicModel object.
        """

        inst = cls(year=data["year"], month=data["month"], model=model)

        inst._hashes = data.get("hashes", {})
        inst.prepared
        return inst

    def to_dict(self) -> dict:
        """Convert the AtomicModel to a dictionary.

        Returns:
            dict: Dictionary with the model data.
        """

        return {
            "year": self.year,
            "month": self.month,
            "ref_path": str(self.ref_path),
            "path": str(self.path),
            "hashes": self._hashes,
            "prepared": self.prepared,
        }

    def dump(self):
        """Dump the model result to a file.

        Returns:
            dict: Dictionary with the model data.
        """
        info = self.to_dict()

        with open(self.path / "meta.json", "w") as f:
            json.dump(info, f, indent=4)
        logger.info("Model result dumped to %s", self.path / "meta.json")


class BaseModel(abc.ABC):
    """Base class for geospatial modeling.

    Args:
        name (str): The name of the model.
        source (BaseDataset): The source of the model.
        interpolate (bool, optional): Interpolate the source to the same grid as the target. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the model.
    """

    SUPPORTED_WEATHER_DATA_CONFIGS: tuple[str]
    metadata_keys: set[str] = {
        "name",
        "module",
        "years",
        "months",
        "files_orig",
        "files_prepared",
        "weather_data_config",
    }

    def __init__(self, source: BaseDataset, **kwargs):
        if not isinstance(source, BaseDataset):
            raise ValueError(f"Source must be a Dataset, but got {type(source)}.")
        if source.weather_config not in self.SUPPORTED_WEATHER_DATA_CONFIGS:
            raise ValueError(
                f"Weather data config {source.weather_config} is not supported by this model."
            )

        if not source.downloaded:
            raise ValueError("The source Dataset for this model is not prepared.")

        self.source = source
        self._extra_kwargs = kwargs
        self._prepared = False

        self._ref_path = model_dir.parent / self.source.module
        self._results = self._prepare_results()

    def __repr__(self):
        return f"Model(source={self.source}, type={self.type})"

    @property
    def frequency(self) -> str:
        """Frequency of the model."""
        return self.source.frequency

    @property
    @abc.abstractmethod
    def type(self) -> str:
        """Type of the model."""

    def _prepare_results(self) -> dict[int, dict[int, ModelResult]]:
        """Prepare the results of the model.

        Returns:
            dict: Dictionary with the results of the model.
        """

        years = list(range(self.source.years.start, self.source.years.stop + 1))
        months = list(range(self.source.months.start, self.source.months.stop + 1))

        results: dict[int, dict[int, ModelResult]] = {}
        for year in years:
            results[year] = {}
            for month in months:
                results[year][month] = ModelResult.from_year_month(self, year, month)
                results[year][month].path.mkdir(parents=True, exist_ok=True)

        return results

    @property
    def results(self):
        """Get the results of the model.

        Returns:
            dict: Dictionary with the results of the model.
        """
        return self._results

    @property
    def flattened_results(self) -> list[ModelResult]:
        """Flatten the results of the model.

        Returns:
            list: List of ModelResult objects.
        """

        return [
            self._results[year][month]
            for year in self._results
            for month in self._results[year]
        ]

    def estimate(
        self,
        height: int,
        years: slice,
        months: Optional[slice] = None,
        xs: Optional[slice] = None,
        ys: Optional[slice] = None,
        use_real_data: bool = False,
    ) -> xr.DataArray:
        """Estimate the wind speed at given coordinates.

        Args:
            height (int): Height of the wind speed, need to be greater than 0.
            years (slice): Years.
            months (slice, optional): Months. If None, all months are estimated.
            xs (slice): X coordinates. If None, all x coordinates in source are estimated.
            ys (slice): Y coordinates. If None, all y coordinates in source are estimated.
            use_real_data (bool, optional): If available, use real data for estimation. Defaults to False.

        Returns:
            xr.DataArray: Dataset with wind speed.
        """

        return self._estimate_dataset(
            height=height,
            years=years,
            months=months,
            xs=xs,
            ys=ys,
            use_real_data=use_real_data,
        )

    @property
    def prepared(self) -> bool:
        """Check if the model is prepared.

        Returns:
            bool: True if prepared.
        """

        if not self._prepared:
            self._prepared = self._check_prepared()
        return self._prepared

    def _check_prepared(self) -> bool:
        for year in self._results:
            for month in self._results[year]:
                if not self._results[year][month].prepared:
                    return False
        return True

    def prepare(self, force: bool = False):
        """Prepare the model.

        Args:
            force (bool, optional): Force re-prepare the model. Defaults to False.
        """

        self._prepared = False  # NOTE: force re-checking preparedness here
        if self.prepared and not force:
            logger.info("The model is already prepared.")
            return

        for result in self.flattened_results:
            if not result.prepared:
                shutil.rmtree(result.path, ignore_errors=True)
                result.path.mkdir(parents=True, exist_ok=True)

                for ref in tqdm(result.ref_params):
                    if not ref.exists():
                        raise FileNotFoundError(f"Reference file {ref} does not exist.")
                    ref_ds = xr.open_dataset(ref)
                    prepared_ds = self._prepare_dataset(ref_ds)
                    ref_ds.close()
                    result.register(prepared_ds)

            result.dump()
        logger.info("Model prepared successfully.")

    @abc.abstractmethod
    def _prepare_dataset(self, source: xr.Dataset) -> xr.Dataset:
        """Prepare the parameters of a specific source dataset file.

        Args:
            source (xr.Dataset): Source dataset.

        Returns:
            xr.Dataset: Prepared parameter dataset.
        """

    @abc.abstractmethod
    def _estimate_dataset(
        self,
        height: int,
        years: slice,
        months: Optional[slice] = None,
        xs: Optional[slice] = None,
        ys: Optional[slice] = None,
        use_real_data: Optional[bool] = False,
    ) -> xr.DataArray:
        """Estimate the wind speed from a dataset.

        Args:
            height (int): Height of the wind speed, need to be greater than 0.
            years (slice): Years.
            months (slice, optional): Months. If None, all months are estimated.
            xs (slice): X coordinates. If None, all x coordinates in source are estimated.
            ys (slice): Y coordinates. If None, all y coordinates in source are estimated.
            use_real_data (bool, optional): If available, use real data for estimation. Defaults to False.

        Returns:
            xr.DataArray: Dataset with wind speed.
        """
