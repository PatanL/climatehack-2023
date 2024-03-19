import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
import h5py
import hdf5plugin
from datetime import datetime, time, timedelta
import numpy as np

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
            with h5py.File(file, 'r') as f:
                l = len(f['pv'])
                self.length += l
                self.individual_lens.append(l)

    def __len__(self):
        return self.length - 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} is out of bounds for dataset of length {self.length}")
        
        # find which file and local index this global idx maps to
        cumsum = 0
        for file_idx, length in enumerate(self.individual_lens):
            if idx < cumsum + length:
                break
            cumsum += length
        else:
            raise IndexError("Failed to locate file for index")
        
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
class ShuffleDataset(torch.utils.data.IterableDataset):
  def __init__(self, dataset, buffer_size):
    super().__init__()
    self.dataset = dataset
    self.buffer_size = buffer_size
  def __len__(self):
     return len(self.dataset)
  def __iter__(self):
    shufbuf = []
    try:
      dataset_iter = iter(self.dataset)
      for i in range(self.buffer_size):
        shufbuf.append(next(dataset_iter))
    except:
      self.buffer_size = len(shufbuf)

    try:
      while True:
        try:
          item = next(dataset_iter)
          evict_idx = random.randint(0, self.buffer_size - 1)
          yield shufbuf[evict_idx]
          shufbuf[evict_idx] = item
        except StopIteration:
          break
      while len(shufbuf) > 0:
        yield shufbuf.pop()
    except GeneratorExit:
      pass