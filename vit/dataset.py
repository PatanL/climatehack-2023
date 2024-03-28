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
import h5py
import time as tfs
import os
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

      
import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

import cartopy.crs as ccrs
import numpy as np
import xarray as xr

# Assuming the rest of the ChallengeDataset class remains the same
    
class ChallengeDataset(Dataset):
    def __init__(self, hdf5_file, site_locations, metadata, nwp_file_path, hrv_file_path, sites=None):
        self.hdf5_file = hdf5_file
        self._site_locations = site_locations
        self.metadata = metadata
        self.nwp_file_path = nwp_file_path
        self.hrv_file_path = hrv_file_path

        with h5py.File(self.hdf5_file, 'r') as file:
            available_keys = list(file['pv'].keys())
            self._sites = [site for site in (sites if sites else site_locations["nonhrv"].keys()) if f'data_{site}' in available_keys]

            self.individual_lens = [len(file['pv'][f'data_{site}']) for site in self._sites]
            self.len = sum(self.individual_lens)

    def __len__(self):
        return self.len

    def compute_file_lens(self, filenames):
        lens = []
        for i in range(1, 13):
            file = filenames + f"{i}.parquet"
            pv = self.open_parquet(file, i-1)
            pv = pv[pv.index.get_level_values('timestamp').minute == 0]
            lens.append(len(pv.index))
        return lens

    @lru_cache(None)
    def open_parquet(self, filename, file_idx=-1):
        if file_idx == -1:
            return pd.read_parquet(filename).drop("generation_wh", axis=1)
        else:
            start, stop = month_to_times[file_idx + 1]
            df = pd.read_parquet(filename).drop("generation_wh", axis=1)
            return df[(df.index.get_level_values('timestamp').hour >= start) & (df.index.get_level_values('timestamp').hour <= stop)]

    @lru_cache(None)
    def open_xarray(self, filename, drop=True):
        ds = xr.open_dataset(filename, engine="zarr", consolidated=True, chunks={"time": "auto"})
        return ds.where(ds['time'].dt.minute == 0, drop=True) if drop else ds

    @lru_cache(None)
    def find_file_idx(self, idx):
        if idx < 0 or idx >= self.len:
            raise IndexError("Index out of bounds for dataset length")
        
        cumsum = 0
        for file_idx, length in enumerate(self.individual_lens):
            if idx < cumsum + length:
                return file_idx, idx - cumsum
            cumsum += length
        raise IndexError("Failed to locate file for index")

    def get_nwp_features(self, nwp_data, nwp_coords, nwp_hours):
        NWP_FEATURES = ["t_500", "clcl", "alb_rad", "tot_prec", "ww", "relhum_2m", "h_snow", "aswdir_s", "td_2m", "omega_1000"]
        cropped_features = {}
        for feature in NWP_FEATURES:
            feature_data = nwp_data[feature].sel(time=nwp_hours)
            lat_min, lat_max, lon_min, lon_max = nwp_coords
            cropped_feature_data = feature_data.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
            cropped_features[feature] = cropped_feature_data.to_numpy()
        return cropped_features

    def crop_hrv_data(self, hrv_data, coords):
        x_min, x_max, y_min, y_max = coords
        try:
            cropped_hrv = hrv_data[:, y_min:y_max, x_min:x_max, :]
            expected_shape = (hrv_data.shape[0], y_max - y_min, x_max - x_min, hrv_data.shape[3])
            if cropped_hrv.shape != expected_shape:
                raise ValueError("Cropped HRV data shape is incorrect")
            return cropped_hrv.to_numpy()
        except Exception as e:
            return None

    def __getitem__(self, idx):
        try:
            file_idx, local_idx = self.find_file_idx(idx)
            site_key = f'data_{self._sites[file_idx]}'

            with h5py.File(self.hdf5_file, 'r') as file:
                if site_key not in file['pv'] or site_key not in file['y']:
                    raise ValueError("Data not found for site_key")

                pv_features = np.array(file['pv'][site_key][local_idx])
                pv_targets = np.array(file['y'][site_key][local_idx])

                if 'nonhrv' not in self._site_locations:
                    raise ValueError("Non-HRV satellite coordinates category not found in site_locations")
                
                new_key = int(site_key[5:])
                if new_key not in self._site_locations['nonhrv']:
                    raise ValueError("No satellite coordinates found for site_key within 'nonhrv' category")

                sat_coords = self._site_locations['nonhrv'][new_key]
                hrv_data = self.open_xarray(self.hrv_file_path)
                center_x, center_y = sat_coords
                coords = (center_x - 64, center_x + 63, center_y - 64, center_y + 63)
                hrv_features = self.crop_hrv_data(hrv_data["data"], coords)

                if None in [pv_features, hrv_features, pv_targets]:
                    raise ValueError("One or more return values are None")

                return pv_features, hrv_features, pv_targets
        except Exception as e:
            return None

