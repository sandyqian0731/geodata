"""Offline correctness tests for the mask fixes on this branch.

Four repro vectors over small synthetic GeoTIFFs (in-memory or in temporary
directories; no network, no downloads):

- ``filter_raster`` with combined min/max bounds must filter against the
  ORIGINAL data (the historical boolean chaining passed every pixel);
- ``trim_raster`` must keep the last data row/column (the historical
  inclusive-stop indices fed end-exclusive Window slices);
- ``merge_layer(method="and")`` with a fully excluded (all-zero) layer must
  stay zero (the historical uniform-window heuristic resurrected excluded
  land);
- ``merge_layer(method="sum")`` of two all-one layers must be 2 (the
  historical heuristic discarded the accumulated values);
- ``add_layer`` must leave the source GeoTIFF's nodata tag and pixels
  untouched on disk (the historical ``r+`` open persisted ``nodata = 0``
  into the user's file) while masking the original fill values to 0 in the
  in-memory layer;
- ``add_shape_layer`` must clear the ``saved`` flag so a subsequent
  ``save_mask()`` actually writes (historically it silently no-opped).

The correctness vectors fail on the unpatched module and pass with this
branch. The file is
pytest-style but also directly executable without pytest:

    python tests/pr/mask/test_mask_correctness.py
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio as ras
from rasterio.transform import from_origin

from geodata.mask import Mask, create_temp_tif, filter_raster, trim_raster

_TRANSFORM = from_origin(100.0, 10.0, 0.1, 0.1)


def _write_tif(path, arr, nodata=None, crs="EPSG:4326", transform=_TRANSFORM):
    """Write a small single-band GeoTIFF and return its path."""
    arr = np.asarray(arr)
    with ras.open(
        str(path),
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr, 1)
    return str(path)


def test_filter_raster_min_and_max_bound_combined():
    """min_bound and max_bound together must AND against the ORIGINAL data.

    Elevation vector [100, 250, 5000, 3000, 8000, 50] with min_bound=200 and
    max_bound=4000 must yield [0, 1, 0, 1, 0, 0]; the historical boolean
    chaining bug returned [1, 1, 1, 1, 1, 1].
    """
    elevation = np.array([[100.0, 250.0, 5000.0, 3000.0, 8000.0, 50.0]])
    raster = create_temp_tif(elevation, _TRANSFORM)

    binarized = filter_raster(raster, min_bound=200, max_bound=4000, binarize=True)
    assert np.array_equal(
        binarized.read(1).flatten(), np.array([0, 1, 0, 1, 0, 0])
    ), f"binarized filter returned {binarized.read(1).flatten().tolist()}"

    # binarize=False must keep the original values of the passing pixels
    # (the historical bug returned the raster completely unchanged).
    unbinarized = filter_raster(raster, min_bound=200, max_bound=4000, binarize=False)
    assert np.array_equal(
        unbinarized.read(1).flatten(),
        np.array([0.0, 250.0, 0.0, 3000.0, 0.0, 0.0]),
    ), f"unbinarized filter returned {unbinarized.read(1).flatten().tolist()}"


def test_trim_raster_keeps_last_data_row_and_column():
    """The docstring example must trim to shape (5, 3), keeping the last
    nonzero row (row 4) and column (column 4).

    The historical off-by-one fed inclusive stop indices into end-exclusive
    Window slices, cropping away the last data row/column -> shape (4, 2).
    """
    arr = np.array(
        [
            [0, 0, 9, 0, 0, 0, 0],
            [0, 0, 1, 2, 0, 0, 0],
            [0, 0, 2, 3, 4, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    raster = create_temp_tif(arr, _TRANSFORM)
    trimmed = trim_raster(raster).read(1)

    assert trimmed.shape == (5, 3), f"trimmed shape is {trimmed.shape}, expected (5, 3)"
    assert np.array_equal(trimmed, arr[0:5, 2:5]), f"trimmed content is\n{trimmed}"
    # The '4' (row 2, col 4) and the row-4 '1' must survive the trim.
    assert 4 in trimmed
    assert trimmed[4].any()


def test_trim_raster_all_zero_returned_unchanged():
    """An all-zero raster has no data bounds to trim to: it must come back
    unchanged instead of producing a degenerate window."""
    arr = np.zeros((4, 5), dtype=np.uint8)
    raster = create_temp_tif(arr, _TRANSFORM)
    trimmed = trim_raster(raster).read(1)
    assert trimmed.shape == (4, 5), f"all-zero raster reshaped to {trimmed.shape}"


def test_merge_layer_and_all_zero_layer_stays_zero():
    """AND with a fully-excluded (all-zero) layer must return all zeros.

    The historical len(np.unique)==1 first-write heuristic fired on the
    uniform zero window and copied the next layer wholesale, resurrecting
    excluded land (result all ones).
    """
    tmp = tempfile.mkdtemp(prefix="geodata_mask_and_")
    zeros = np.zeros((4, 4), dtype=np.uint8)
    ones = np.ones((4, 4), dtype=np.uint8)

    m = Mask(
        "test_and",
        layer_path={
            "excluded": _write_tif(Path(tmp) / "excluded.tif", zeros),
            "available": _write_tif(Path(tmp) / "available.tif", ones),
        },
        mask_dir=os.path.join(tmp, "masks"),
    )
    merged = m.merge_layer(method="and", show_raster=False).read(1)
    assert (merged == 0).all(), f"AND resurrected excluded pixels:\n{merged}"

    # Control: AND of two mixed binary layers is the elementwise AND.
    tmp2 = tempfile.mkdtemp(prefix="geodata_mask_and2_")
    a = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    b = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    m2 = Mask(
        "test_and_mixed",
        layer_path={
            "a": _write_tif(Path(tmp2) / "a.tif", a),
            "b": _write_tif(Path(tmp2) / "b.tif", b),
        },
        mask_dir=os.path.join(tmp2, "masks"),
    )
    merged2 = m2.merge_layer(method="and", show_raster=False).read(1)
    assert np.array_equal(
        merged2, np.array([[1, 0], [0, 0]])
    ), f"AND of mixed layers returned\n{merged2}"


def test_merge_layer_sum_of_two_ones_is_two():
    """SUM of two all-one layers (default weights of 1) must be 2 everywhere.

    The historical heuristic treated the uniform already-merged window as a
    first write and discarded the accumulated values, returning 1.
    """
    tmp = tempfile.mkdtemp(prefix="geodata_mask_sum_")
    ones = np.ones((4, 4), dtype=np.uint8)

    m = Mask(
        "test_sum",
        layer_path={
            "first": _write_tif(Path(tmp) / "first.tif", ones),
            "second": _write_tif(Path(tmp) / "second.tif", ones),
        },
        mask_dir=os.path.join(tmp, "masks"),
    )
    merged = m.merge_layer(method="sum", show_raster=False).read(1)
    assert (merged == 2).all(), f"SUM of two all-one layers returned\n{merged}"


def test_add_layer_leaves_source_nodata_untouched():
    """add_layer must never mutate the user's file; fills mask to 0 in memory.

    Historically ``_add_layer`` opened the source with ``r+`` and persisted
    ``nodata = 0`` into the GeoTIFF header on disk, so -9999-style fills were
    re-labeled as valid data for every other tool, and the fills themselves
    leaked through filters (e.g. ``-9999 < max_bound`` passed an elevation
    ceiling).
    """
    tmp = tempfile.mkdtemp(prefix="geodata_mask_nodata_")
    elev = np.array(
        [[100.0, 250.0, -9999.0], [3000.0, -9999.0, 50.0], [0.0, 800.0, 1200.0]],
        dtype=np.float32,
    )
    src_path = _write_tif(Path(tmp) / "elev.tif", elev, nodata=-9999.0)

    m = Mask("test_nodata", layer_path=src_path, layer_name="elev",
             mask_dir=os.path.join(tmp, "masks"))

    # The file on disk is untouched: nodata tag and pixel values unchanged.
    with ras.open(src_path) as reopened:
        assert reopened.nodata == -9999.0, (
            f"source nodata was rewritten on disk to {reopened.nodata}"
        )
        assert (reopened.read(1) == elev).all(), "source pixels were mutated"

    # The in-memory layer follows the module convention: fills masked to 0.
    band = m.layers["elev"].read(1)
    assert band[0, 2] == 0 and band[1, 1] == 0, (
        f"-9999 fills were not masked to 0 in the layer:\n{band}"
    )
    # Real values survive.
    assert band[0, 0] == 100.0 and band[2, 2] == 1200.0


def test_add_shape_layer_clears_saved_flag():
    """add_shape_layer must invalidate ``saved`` so save_mask() writes.

    Every other layer-mutating method clears the flag; historically
    ``add_shape_layer`` did not, so ``save_mask()`` after adding shapes
    logged success while writing nothing.
    """
    import shapely.geometry

    tmp = tempfile.mkdtemp(prefix="geodata_mask_shape_saved_")
    ones = np.ones((4, 4), dtype=np.uint8)
    m = Mask("test_shape_saved",
             layer_path=_write_tif(Path(tmp) / "base.tif", ones),
             layer_name="base",
             mask_dir=os.path.join(tmp, "masks"))
    m.save_mask()
    assert m.saved is True, "precondition: mask saved"

    box = shapely.geometry.box(100.05, 9.75, 100.25, 9.95)
    m.add_shape_layer({"box": box}, reference_layer="base")
    assert m.saved is False, (
        "add_shape_layer left saved=True; save_mask() would silently no-op"
    )


if __name__ == "__main__":
    import traceback

    all_tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = []
    for test_name, test_fn in all_tests:
        try:
            test_fn()
            print(f"PASS {test_name}")
        except Exception:
            failed.append(test_name)
            print(f"FAIL {test_name}")
            traceback.print_exc()
    print(f"\n{len(all_tests) - len(failed)}/{len(all_tests)} tests passed")
    sys.exit(1 if failed else 0)
