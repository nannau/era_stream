This simple project is an implementation of this really nice [blog post](https://earthmover.io/blog/cloud-native-dataloader) by the folks over at Earthmover. I've made this ERA5 dataset to use [Google's ARCO ERA5 catalog](https://cloud.google.com/storage/docs/public-datasets/era5), and have added the ability to use xESMF to re-grid ERA5 data to a target grid on the fly.

#### Configuration

```python
import dask
from torch.utils.data import Dataset, DataLoader

from era_stream import ERA5Dataset

dask.config.set(scheduler="threads", num_workers=4)


dataset = ERA5Dataset(
    file_path="gs://gcp-public-data-arco-era5/ar/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr",
    variables=['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'surface_pressure', 'total_precipitation'],
    patch_size={"latitude": 48, "longitude": 48, "time": 3},
    regrid=True,
    regrid_file="bilinear_181x360_88x80.nc"
)

training_generator = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=32,
    prefetch_factor=3,
    persistent_workers=True,
    multiprocessing_context=”forkserver”
)
```