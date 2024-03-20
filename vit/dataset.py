import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
import h5py
import hdf5plugin
from datetime import datetime, time, timedelta
import numpy as np
from functools import lru_cache
from tqdm import tqdm
import xarray as xr
import pandas as pd
import sys

BATCH_SIZE = 32
class HDF5Dataset(Dataset):
    def __init__(self, files, pv, sat, nwp, extra):
        self.files = files
        self.pv = pv
        self.sat = sat
        self.nwp = nwp
        self.extra = extra
        self.length = 0
        self.individual_lens = []
        # Open the file quickly to get the number of keys
        for file in self.files:
            print(f"Opening file {file}.")
            try:
                with h5py.File(file, 'r') as f:
                    l = len(f['pv'])
                    self.length += l - 1
                    self.individual_lens.append(l)
            except Exception as e:
               print(e)
               pass
        print("Warming up the dataloader!")
        for i in tqdm(range(self.length)):
            self.find_file_idx(i)

    def __len__(self):
        return self.length
    @lru_cache(None)
    def find_file_idx(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} is out of bounds for dataset of length {self.length}")
        
        # find which file and local index this global idx maps to
        cumsum = 0
        for file_idx, length in enumerate(self.individual_lens):
            if idx < cumsum + length:
                idx -= cumsum
                break
            cumsum += length
        else:
            raise IndexError("Failed to locate file for index")
        return file_idx, idx
    def __getitem__(self, idx):
        file_idx, idx = self.find_file_idx(idx)
        with h5py.File(self.files[file_idx], 'r') as f:
            data_name = f'data_{idx}'
            data = []
            
            if self.pv:
                data.append(torch.from_numpy(f['pv'][data_name][...]))
            if self.sat:
                data.append(torch.from_numpy(f['nonhrv'][data_name][...]))
            if self.nwp:
                data.append(torch.from_numpy(f['nwp'][data_name][...]))
            if self.extra:
                data.append(torch.from_numpy(f['extra'][data_name][...]))
                
            data.append(torch.from_numpy(f['y'][data_name][...]))
            return data

month_to_times = {
    1: (8, 16),
    2: (8, 17),
    3: (7, 18),
    4: (7, 19),
    5: (6, 20),
    6: (5, 20),
    7: (5, 20),
    8: (6, 20),
    9: (7, 19),
    10: (7, 18),
    11: (7, 16),
    12: (8, 16)
}

def l_shuffle(pvs, hrvs, weathers):
   p = np.random.permutation(len(pvs))
   return list(np.array(pvs)[p]), list(np.array(hrvs)[p]), list(np.array(weathers)[p]), p

      
class ChallengeDataset(Dataset):
    def __init__(self, pvs, hrvs, weathers, site_locations, metadata, sites=None):
        self.time_index = []
        self.pv, self.sat, self.weather, = [pvs + f"{i}.parquet" for i in range(1, 13)], [hrvs + f"{i}.zarr" for i in range(1, 13)], [weathers + f"{i}.zarr" for i in range(1, 13)]
        self.individual_lens = self.compute_file_lens(pvs)
        self._site_locations = site_locations
        self._sites = sites if sites else list(site_locations["nonhrv"].keys())
        self.metadata = metadata
        self.len = sum(self.individual_lens)
        self.num_sites = len(self._sites)
        self.pv_metadata_file = "/data/pv/metadata.csv"

    def __len__(self):
        return self.len

    def compute_file_lens(self, filenames):
        lens = []
        for i in range(1, 13):
            file = filenames + f"{i}.parquet"
            pv = self.open_parquet(file, i-1)
            pv = pv[pv.index.get_level_values('timestamp').minute == 0]
            help = list(pv.index)
            self.time_index.append(help)
            lens.append(len(help))
        return lens

    @lru_cache(None)
    def open_parquet(self, filename, file_idx=-1):
        if file_idx == -1:
            return pd.read_parquet(filename).drop("generation_wh", axis=1)
        else:
            start, stop = month_to_times[file_idx + 1]
            df = pd.read_parquet(filename).drop("generation_wh", axis=1)
            df = df[df.index.get_level_values('timestamp').hour >= start]
            return df[df.index.get_level_values('timestamp').hour <= stop]

    @lru_cache(None)
    def open_xarray(self, filename, drop=True):
        ds = xr.open_dataset(
            filename,
            engine="zarr",
            consolidated=True,
            chunks={"time": "auto"}
        )
        if drop:
            return ds.where(ds['time'].dt.minute == 0, drop=True)
        else:
            return ds

    @lru_cache(None)
    def find_file_idx(self, idx):
        if idx < 0 or idx >= self.len:
            raise IndexError(f"Index {idx} is out of bounds for dataset of length {self.length}")
        
        # find which file and local index this global idx maps to
        cumsum = 0
        for file_idx, length in enumerate(self.individual_lens):
            if idx < cumsum + length:
                idx -= cumsum
                break
            cumsum += length
        else:
            raise IndexError("Failed to locate file for index")
        return file_idx, idx

    def __getitem__(self, idx):
        # idx 0 is site 0 timestep 0, idx 1 is site 1 timestep 0, and so on...
        file_idx, idx = self.find_file_idx(idx)
        filename = self.sat[file_idx]

        timestep, site = self.time_index[file_idx][idx]
        time = timestep.to_pydatetime().replace(tzinfo=None)
        first_hour = slice(str(time), str(time + timedelta(minutes=55)))
        # get pv features and target 
        pv = self.open_parquet(self.pv[file_idx], file_idx)
        pv_features = pv.xs(first_hour, drop_level=False).xs(site, level=1).to_numpy().squeeze(-1)
        pv_targets = pv.xs(
            slice(  # type: ignore
                str(time + timedelta(hours=1)),
                str(time + timedelta(hours=4, minutes=55)),
            ),
            drop_level=False,
        ).xs(site, level=1).to_numpy().squeeze(-1)

        # sat data
        hrv = self.open_xarray(filename, drop=False)
        hrv_data = hrv["data"].sel(time=first_hour).to_numpy().transpose(3, 0, 1, 2)
        x, y = self._site_locations["nonhrv"][site]
        hrv_features = hrv_data[:, :, y - 64 : y + 64, x - 64 : x + 64]

        # weather data
        weather = self.open_xarray(self.weather[file_idx], drop=False)
        x, y = self._site_locations["weather"][site]
        weather_features = np.squeeze(weather.sel(time=first_hour).to_array().to_numpy())
        weather_features = weather_features[:, y - 64 : y + 64, x - 64 : x + 64]

        EXTRA_FEATURES = ["latitude_rounded", "longitude_rounded", "orientation", "tilt"]
        # get extra data 
        with open(self.pv_metadata_file, "r") as f:
            pv_metadata = pd.read_csv(f)
            pv_metadata.set_index("ss_id", inplace=True)
            extra = pv_metadata.loc[site, EXTRA_FEATURES].to_numpy().astype(np.float32)

        return pv_features, hrv_features, weather_features, extra, pv_targets
