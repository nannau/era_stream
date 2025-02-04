# "ERA" Stream
This simple project is an implementation of this really nice [blog post](https://earthmover.io/blog/cloud-native-dataloader) by the folks over at Earthmover. It leans heavily on [the code they provide]([url](https://github.com/earth-mover/dataloader-demo/tree/main)). The only difference between this implementation, and Earthmover's, is that I've simplified the `dataset` object and made the following a bit more explicit:

1. simply apply transforms on the `xarray.Dataset` object, and
2.  allow for the use of `torchvision.transforms.Compose` objects.

The project is still under development, and I'm still learning from the Earthmover code at the time of writing!

This code was tested with [ARCO data provided by Google](https://cloud.google.com/storage/docs/public-datasets/era5) and had very nice results right out of the box with very little latency. It should also work with other cloud storage objects. For example, in the coming weeks/months I will be working on implementing this with high-resolution WRF model output.

### Installation

1. Clone the repository
2. Install with pip `pip install .` in a fresh venv

### Example

Please see `example.py` for a more complete example of how I've used the object so far. 

Below is a simple setup example:

```python
import xarray as xr
from torch.utils.data import DataLoader

from era_stream import ERA5Dataset

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

# this waits for the first batch to be loaded
_ = next(iter(training_generator))

num_epochs = 10
num_batches = len(training_generator)

for epoch in range(num_epochs):
    for i, sample in enumerate(training_generator):
        ...
```
