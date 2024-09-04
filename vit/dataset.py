import torch
from torch.utils.data import Dataset, IterableDataset
import h5py
import hdf5plugin
from datetime import datetime, timedelta, time
import numpy as np
from functools import lru_cache
from tqdm import tqdm
import xarray as xr
import pandas as pd
import h5py
import time as timer
BATCH_SIZE = 32
NWP_FEATURES = [
    "t_500", "clcl", "alb_rad", "tot_prec", "ww",
    "relhum_2m", "h_snow", "aswdir_s", "td_2m", "omega_1000"
]

class HDF5Dataset(Dataset):
    def __init__(self, files, sat_folder, nwp_folder, pv, sat, nwp, extra, individual_lens):
        self.files = files
        self.file_handles = [None for f in files]
        self.sat_folder = sat_folder
        self.nwp_folder = nwp_folder
        self.pv = pv
        self.sat = sat
        self.nwp = nwp
        self.extra = extra
        self.length = sum(individual_lens)
        self.individual_lens = individual_lens
        
        print("Warming up the dataloader!")
        for i in tqdm(range(self.length)):
            self.find_file_idx(i)

    def __len__(self):
        return self.length
    @lru_cache(None)
    def find_file_idx(self, idx):
        if len(self.files) == 1:
            return 0, idx
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
        f = self.file_handles[file_idx]
        if f == None:
            self.file_handles[file_idx] = h5py.File(self.files[file_idx], 'r', libver='latest', swmr=True)
            f = self.file_handles[file_idx]
        data_name = f'data_{idx}'
        data = []
        datett = datetime.utcfromtimestamp(f['time'][data_name][...][0])
        time = datett # time is a little bit of a misnomer, this is really the start of the one-hour period before the time we are predicting
        if self.pv:
            walltime = timer.time()
            crop = torch.from_numpy(f['pv'][data_name][...])
            try:
                assert crop.shape == torch.Size([12])
            except Exception as e:
                print(crop.shape)
                return None
            # print("pv: ", timer.time() - walltime)
            data.append(crop)
        if self.sat:
            walltime = timer.time()
            x, y = f['nonhrv'][data_name][...]
            x, y = int(x), int(y)
            # read from our horrible numpy method
            try:
                dt = time
                base_folder = self.sat_folder
                folder_name = base_folder + dt.strftime('%y-%m-%d') + "/"
                file_name = dt.strftime('%H:%M:%S') + ".npy"
                total_name = folder_name + file_name
                try:
                    crop = np.load(total_name)
                except: 
                    return None
            except Exception as e:
                return None
            # end reading
            
            crop = torch.from_numpy(crop[:, y - 64 : y + 64, x - 64 : x + 64, :])
            crop = crop.permute((3, 0, 1, 2))
            try:
                assert crop.shape == torch.Size([11, 12, 128, 128])
            except Exception as e:
                return None
            # print("sat: ", timer.time() - walltime)
            data.append(crop)
        if self.nwp:
            walltime = timer.time()
            x, y = f['nwp'][data_name][...]
            x, y = int(x), int(y)
            # read from our horrible numpy method
            try:
                dt = time
                base_folder = self.nwp_folder
                folder_name = base_folder + dt.strftime('%y-%m-%d') + "/"
                file_name = dt.strftime('%H:%M:%S') + ".npy"
                total_name = folder_name + file_name
                crop = np.load(total_name)
            except Exception as e:
                print(e) 
                return None
            # end readin
            crop = torch.from_numpy(crop[:, :, y - 64 : y + 64, x - 64 : x + 64])
            try:
                assert crop.shape == torch.Size([10, 6, 128, 128])
            except Exception as e:
                return None
            # print("nwp: ", timer.time() - walltime)
            data.append(crop)
        if self.extra:
            walltime = timer.time()
            data.append(torch.from_numpy(f['extra'][data_name][...]))
            # print("extra: ", timer.time() - walltime)
        walltime = timer.time()
        y = torch.from_numpy(f['y'][data_name][...])
        try:
            assert y.shape == torch.Size([48])
        except:
            return None
        # print("y: ", timer.time() - walltime)
        data.append(y)
        return data

month_to_times = {
    1: (time(8), time(16)),
    2: (time(8), time(17)),
    3: (time(7), time(18)),
    4: (time(7), time(19)),
    5: (time(6), time(20)),
    6: (time(5), time(20)),
    7: (time(5), time(20)),
    8: (time(6), time(20)),
    9: (time(7), time(19)),
    10: (time(7), time(18)),
    11: (time(7), time(16)),
    12: (time(8), time(16))
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
    

class ChallengeDataset(IterableDataset):
    def __init__(self, pv, hrv, weather, site_locations, metadata, sites=None):
        self.pv = pv
        self.hrv = hrv
        self.weather = weather
        self._site_locations = site_locations
        self._sites = sites if sites else list(site_locations["hrv"].keys())
        self.metadata = metadata
        self.len = 0
        for _ in self._get_image_times():
            self.len += 1
        

    def __len__(self):
        return self.len

    def _get_image_times(self):
        min_date = datetime(2020, 1, 1)
        max_date = datetime(2020, 12, 31)

        date = min_date
        while date <= max_date:
            month = date.month
            start_time, end_time = month_to_times[month]
            current_time = datetime.combine(date, start_time)
            while current_time.time() < end_time:
                if current_time:
                    yield current_time

                current_time += timedelta(minutes=60)

            date += timedelta(days=1)

    def __iter__(self):
        for time in self._get_image_times():
            first_hour = slice(str(time), str(time + timedelta(minutes=55)))

            pv_features = self.pv.xs(first_hour, drop_level=False)  # type: ignore
            pv_targets = self.pv.xs(
                slice(  # type: ignore
                    str(time + timedelta(hours=1)),
                    str(time + timedelta(hours=4, minutes=55)),
                ),
                drop_level=False,
            )

            hrv_data = self.hrv["data"].sel(time=first_hour).to_numpy()

            for site in self._sites:
                try:
                    # Get solar PV features and targets
                    site_features = pv_features.xs(site, level=1).to_numpy().squeeze(-1)
                    site_targets = pv_targets.xs(site, level=1).to_numpy().squeeze(-1)
                    assert site_features.shape == (12,) and site_targets.shape == (48,)

                    # Get a 128x128 HRV crop centred on the site over the previous hour
                    x, y = self._site_locations["hrv"][site]
                    hrv_features = hrv_data[:, y - 64 : y + 64, x - 64 : x + 64, 0]
                    assert hrv_features.shape == (12, 128, 128)

                    # weather
                    nwp_features_arr = []
                    nan_nwp = False
                    x_nwp, y_nwp = self._site_locations["weather"][site]
                    for feature in ["t_500", "clcl", "alb_rad", "tot_prec", "ww", "relhum_2m", "h_snow", "aswdir_s", "td_2m", "omega_1000"]:
                        data = self.weather[feature].sel(time=first_hour).to_numpy()
                        if data.shape[0] != 6 or np.isnan(data).any():
                            nan_nwp = True
                            break
                        data = data[:, y_nwp - 64 : y_nwp + 64, x_nwp - 64 : x_nwp + 64]
                        assert data.shape == (6, 128, 128)
                        nwp_features_arr.append(data)
                    if nan_nwp:
                        continue
                    nwp = np.stack(nwp_features_arr, axis=0)

                    # extra data
                    extra = self.metadata.loc[site, ["latitude_rounded", "longitude_rounded", "orientation", "tilt"]].to_numpy().astype(np.float32)
                    assert extra.shape == (4,)
                    # How might you adapt this for the non-HRV, weather and aerosol data?
                except:
                    continue

                yield site_features, hrv_features, nwp, extra, site_targets