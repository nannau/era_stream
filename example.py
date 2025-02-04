import time

from torch.utils.data import DataLoader
import dask
from dask.cache import Cache
import xarray as xr
from era_stream import ERA5Dataset


# comment these the next two lines out to disable Dask's cache
cache = Cache(1e10)  # 10gb cache
cache.register()


if __name__ == "__main__":

    dask.config.set(scheduler="threads", num_workers=4)

    variables = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "surface_pressure",
        "total_precipitation",
    ]
    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr",
        consolidated=True,
    )[variables]

    # select a subset of the data
    ds = ds.sel(time=slice("2010-01-01", "2010-01-02"))

    # Add any other xarray preprocessing steps here, e.g. regridding, etc.
    # pro tip: use xesmf regridding files for fast regridding

    # if you have a transform that can be applied torchvision style, you can also pass this to 
    # the ERA5Dataset class by composing a transrom with torchvision.transforms.Compose

    dataset = ERA5Dataset(
        dataset=ds,
        patch_size={"latitude": 48, "longitude": 48, "time": 1},
    )

    training_generator = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        prefetch_factor=3,
        persistent_workers=True,
        multiprocessing_context="forkserver",
    )

    # this force waits for the first batch to be loaded
    _ = next(iter(training_generator))

    num_epochs = 10
    num_batches = len(training_generator)

    for epoch in range(num_epochs):
        for i, sample in enumerate(training_generator):
            tt0 = time.time()
            print("Retrieved batch with shape:", sample.shape)
            time.sleep(2)  # simulate model training
            tt1 = time.time()
            print({"event": "training end", "batch": i, "duration": tt1 - tt0})
            if i == num_batches - 1:
                break

