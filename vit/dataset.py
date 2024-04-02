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

BATCH_SIZE = 32
NWP_FEATURES = [
    "t_500", "clcl", "alb_rad", "tot_prec", "ww",
    "relhum_2m", "h_snow", "aswdir_s", "td_2m", "omega_1000"
]
import dask
dask.config.set(scheduler='synchronous')
class HDF5Dataset(Dataset):
    def __init__(self, files, sat_file, nwp_file, pv, sat, nwp, extra):
        self.files = files
        self.sat_file = xr.open_dataset(sat_file, engine="zarr", chunks={"time": "auto"})
        self.nwp_file = xr.open_dataset(nwp_file, engine="zarr", chunks={"time": "auto"})
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
            datett = datetime.utcfromtimestamp(f['time'][data_name][...][0])
            time = datett # time is a little bit of a misnomer, this is really the start of the one-hour period before the time we are predicting

            if self.pv:
                data.append(torch.from_numpy(f['pv'][data_name][...]))
            if self.sat:
                x, y = f['nonhrv'][data_name][...]
                x, y = int(x), int(y)
                first_hour = slice(str(time), str(time + timedelta(minutes=55)))
                crop = self.sat_file["data"].sel(time=first_hour).to_numpy()[:, y - 64 : y + 64, x - 64 : x + 64, :]
                crop = torch.from_numpy(crop).permute((3, 0, 1, 2))
                data.append(crop)
            if self.nwp:
                x, y = f['nwp'][data_name][...][0]
                x_nwp, y_nwp = int(x), int(y)
                T = time + timedelta(hours=1)
                # Check if time is on the hour or not
                if T.minute == 0:
                    nwp_hours = slice(str(T - timedelta(hours=1)), str(T + timedelta(hours=4)))
                else:
                    nwp_hours = slice(str(T - timedelta(hours=1, minutes=time.minute)), str(T + timedelta(hours=4) - timedelta(minutes=time.minute)))
                nwp_hours = slice(str(T - timedelta(hours=1, minutes=time.minute)), str(T + timedelta(hours=4) - timedelta(minutes=time.minute)))
                nwp_features_arr = []
                for feature in NWP_FEATURES:
                    data2 = self.nwp_file[feature].sel(time=nwp_hours).to_numpy()
                    if data2.shape[0] != 6 or np.isnan(data2).any():
                        return None
                    # 128x128 crop
                    data2 = data2[:, y_nwp - 64 : y_nwp + 64, x_nwp - 64 : x_nwp + 64]
                    assert data2.shape == (6, 128, 128)
                    nwp_features_arr.append(data2)
                nwp = np.stack(nwp_features_arr, axis=0)
                data.append(torch.from_numpy(nwp))
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
        self.pv, self.sat, self.weather, = [pvs + f"{i}.parquet" for i in range(1, 13)], [hrvs + f"{i}.zarr" for i in range(1, 13)], [weathers + f"{i}.hdf5" for i in range(1, 13)]

        self.individual_lens = self.compute_file_lens(pvs)
        self._site_locations = site_locations
        self._sites = sites if sites else list(site_locations["nonhrv"].keys())
        self.metadata = metadata
        self.len = sum(self.individual_lens)
        self.num_sites = len(self._sites)
        self.pv_metadata_file = "/data/pv/metadata.csv"
        self.weather_times = []
        for idx in range(0, 12):
            with h5py.File(self.weather[idx], "r") as hdf_file:
                time2 = np.array(hdf_file['time'])
                time2 = np.array(list(map(lambda x: datetime.utcfromtimestamp(int(x)), time2)))
                self.weather_times.append(time2)
        self.pv_handles = {}
        self.sat_handles = {}
        self.weather_handles = {}
        print("opening files in cache")
        for item in self.sat:
            self.sat_handles[item] = self.open_xarray(item, drop=False)
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

    def open_parquet(self, filename, file_idx=-1):
        if file_idx == -1:
            try:
                return pd.read_parquet(filename).drop("generation_wh", axis=1)
            except:
                return pd.read_parquet(filename)
        else:
            start, stop = month_to_times[file_idx + 1]
            try:
                df = pd.read_parquet(filename).drop("generation_wh", axis=1)
            except:
                df = pd.read_parquet(filename)
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

    def hdf5_sel(self, times, start, stop):
        return np.where((times >= start) & (times <=stop))[0]


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
        # idx 0 is site 0 timestep 0, idx 1 is site 1 timestep 0, and so on...o
        start = tfs.time()
        orig = idx
        file_idx, idx = self.find_file_idx(idx)
        filename = self.sat[file_idx]

        timestep, site = self.time_index[file_idx][idx]
        time = timestep.to_pydatetime().replace(tzinfo=None)
        first_hour = slice(str(time), str(time + timedelta(minutes=55)))
        
        finish_setup = tfs.time()
        print("setup:", finish_setup - start)

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
        try:
            assert pv_features.shape == (12,) and pv_targets.shape == (48,)
        except:
            return self.__getitem__(orig + 1)

        pv = tfs.time()
        print("pv:", pv - finish_setup)
        # sat data
        hrv = self.open_xarray(filename, drop=False)
        hrv_data = hrv["data"].sel(time=first_hour).to_numpy().transpose(3, 0, 1, 2)
        x, y = self._site_locations["nonhrv"][site]
        hrv_features = hrv_data[:, :, y - 64 : y + 64, x - 64 : x + 64]

        try:
            assert hrv_features.shape == (11, 12, 128, 128)
        except:
            return self.__getitem__(orig + 1)
        
        sat = tfs.time()
        print("sat:", sat - pv)
        # weather data
        with h5py.File(self.weather[file_idx], "r") as hdf_file:
            x, y = self._site_locations["weather"][site]
            time2 = self.weather_times[file_idx]
            all_weather_features = 0
            n = self.hdf5_sel(time2, time + timedelta(hours=-1), time + timedelta(hours=4))
            nth_values = []
            SKIP_LIST = set(['latitude', 'longitude', 'time'])
            for name, dataset in hdf_file.items():
                # Check if the item is a dataset
                if name in SKIP_LIST: 
                    continue
                if isinstance(dataset, h5py.Dataset):
                    # Fetch the nth value from the dataset
                    # Ensuring n is within the bounds of the dataset's size
                    if n.all() < len(dataset):
                        nth_value = dataset[n]
                        # Append the nth value as a numpy array
                        nth_values.append(np.array(nth_value)[:, y - 64 : y + 64, x - 64 : x + 64])
                    else:
                        # Handle cases where n exceeds dataset size
                        print(f"Dataset '{name}' does not have a {n}th value.")
            weather_features = np.stack(nth_values, axis=0)
            if type(all_weather_features) == type(0):
                all_weather_features = weather_features
            else:
                all_weather_features = np.stack((all_weather_features, weather_features), axis=1)

        weather = tfs.time()
        # print("weather:", weather - sat)

        EXTRA_FEATURES = ["latitude_rounded", "longitude_rounded", "orientation", "tilt"]
        # get extra data 
        with open(self.pv_metadata_file, "r") as f:
            pv_metadata = pd.read_csv(f)
            pv_metadata.set_index("ss_id", inplace=True)
            extra = pv_metadata.loc[site, EXTRA_FEATURES].to_numpy().astype(np.float32)

        ex = tfs.time()
        # print("extra:", ex - weather)

        return pv_features, hrv_features, all_weather_features, extra, pv_targets
