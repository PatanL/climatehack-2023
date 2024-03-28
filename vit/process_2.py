import os
import multiprocessing
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from ocf_blosc2 import Blosc2
from torch.utils.data import DataLoader, IterableDataset, Dataset, SubsetRandomSampler
from torchinfo import summary
import json
import numpy as np
import h5py
from tqdm import tqdm

plt.rcParams["figure.figsize"] = (20, 12)

# https://www.worlddata.info/europe/united-kingdom/sunset.php
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

def get_image_times(year, month):
    min_date = datetime(year, month, 1)
    
    if month == 2:
        max_date = datetime(year, month, 28)
    elif month in [4, 6, 9, 11]:
        max_date = datetime(year, month, 30)
    else:
        max_date = datetime(year, month, 31)        

    start_time, end_time = month_to_times[month]
    
    date = min_date
    while date <= max_date:
        current_time = datetime.combine(date, start_time)
        
        while current_time.time() < end_time:
            # Drop 92% of times
            if current_time and np.random.rand() < 0.08:
                yield current_time
                
            minutes_to_add = int(np.random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]))
            current_time += timedelta(minutes=minutes_to_add, hours=1)
            
        date += timedelta(days=1)


import pandas as pd
import xarray as xr
import numpy as np
import json
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import json

# Presumed that NWP_FEATURES and EXTRA_FEATURES are defined globally
NWP_FEATURES = ["t_500", "clcl", "alb_rad", "tot_prec", "ww", "relhum_2m", "h_snow", "aswdir_s", "td_2m", "omega_1000"]
EXTRA_FEATURES = ["latitude_rounded", "longitude_rounded", "orientation", "tilt"]


import pandas as pd
import xarray as xr
import numpy as np
import json
from datetime import datetime, timedelta

NWP_FEATURES = ["t_500", "clcl", "alb_rad", "tot_prec", "ww", "relhum_2m", "h_snow", "aswdir_s", "td_2m", "omega_1000"]
EXTRA_FEATURES = ["latitude_rounded", "longitude_rounded", "orientation", "tilt"]

def worker(dates, sat_type, all_sites):  
    with open("./indices.json") as f:
        site_locations = json.load(f)

    pv_metadata_file = "./data/pv/metadata.csv"
    pv_metadata = pd.read_csv(pv_metadata_file)
    pv_metadata.set_index("ss_id", inplace=True)

    for year, month in dates:
        print(f"Processing year {year}, month {month}...")
        pv_file_path = f"./data/pv/{year}/{month}.parquet"
        sat_file_path = f"./data/satellite-{sat_type}/{year}/{month}.zarr.zip"
        nwp_file_path = f"./data/weather/{year}/{month}.zarr.zip"

        pv_data = pd.read_parquet(pv_file_path)

        # Check if 'timestamp' and 'ss_id' are in the index
        if 'timestamp' not in pv_data.index.names or 'ss_id' not in pv_data.index.names:
            continue

        for time in get_image_times(year, month):
            for site in all_sites:
                if site not in pv_metadata.index:
                    continue

                try:
                    pv_features = pv_data.loc[(time, site)]

                    if pv_features.empty:
                        continue

                    print(f"Processing site {site} at time {time}")
                    site_features = pv_features.to_numpy() if isinstance(pv_features, pd.Series) else pv_features.values
                    site_targets = site_features  # Adjust if targets are stored differently

                    x, y = site_locations['satellite'][site]
                    crop_coords_sat = (x - 64, x + 64, y - 64, y + 64)

                    T = time + timedelta(hours=1)
                    nwp_hours = slice(T - timedelta(hours=1), T + timedelta(hours=4))

                    nwp_features_arr = []
                    for feature in NWP_FEATURES:
                        data = nwp_data[feature].sel(time=nwp_hours).to_numpy()
                                        
                        if data.shape[0] != 6 or np.isnan(data).any():
                            print(f"NWP data for feature {feature} is incomplete or contains NaN values at time {time} for site {site}.")
                            continue

                        nwp_features_arr.append(data)

                    x_nwp, y_nwp = site_locations['weather'][site]
                    crop_coords_nwp = (x_nwp - 64, x_nwp + 64, y_nwp - 64, y_nwp + 64)

                    extra = pv_metadata.loc[site, EXTRA_FEATURES].to_numpy()

                    yield (site, site_features, crop_coords_sat, crop_coords_nwp, extra, site_targets, time)
                except KeyError as e:



import pandas as pd
import xarray as xr
import numpy as np
import json
from datetime import datetime, timedelta
import h5py
from tqdm import tqdm

import h5py
from tqdm import tqdm

# Assuming NWP_FEATURES and EXTRA_FEATURES are defined elsewhere in the code as shown previously
NWP_FEATURES = ["t_500", "clcl", "alb_rad", "tot_prec", "ww", "relhum_2m", "h_snow", "aswdir_s", "td_2m", "omega_1000"]
EXTRA_FEATURES = ["latitude_rounded", "longitude_rounded", "orientation", "tilt"]
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
def process_data(sat_type):
    # Load site identifiers from the metadata file
    pv_metadata_file = "./data/pv/metadata.csv"
    pv_metadata = pd.read_csv(pv_metadata_file)
    all_sites = pv_metadata['ss_id'].unique()  # Get all unique site identifiers

    for year in range(2021, 2022):  # Adjust the range as necessary
        for month in range(1, 13):  # Adjust the range as necessary
            file_path = f'./data/processed_data/processed_train_{month}.hdf5'
            with h5py.File(file_path, 'a') as f_train:
                f_pv = f_train.require_group('pv')
                f_coords_sat = f_train.require_group('coords_sat')
                f_coords_nwp = f_train.require_group('coords_nwp')
                f_extra = f_train.require_group('extra')
                f_time = f_train.require_group('time')
                f_y = f_train.require_group('y')

                site_times = {}  # Dictionary to accumulate timestamps for each site

                for site, pv, coords_sat, coords_nwp, extra, y, time in tqdm(worker([(year, month)], sat_type, all_sites)):
                    dataset_name = f'data_{site}'
                    if dataset_name in f_pv:
                        print(f"Data for site {site} already exists, skipping")
                        continue

                    # Accumulate timestamps for each site
                    if site not in site_times:
                        site_times[site] = []
                    site_times[site].append(time.timestamp())

                    # The rest of the data handling
                    f_pv.create_dataset(dataset_name, data=pv, compression="lzf")
                    f_coords_sat.create_dataset(dataset_name, data=coords_sat)
                    f_coords_nwp.create_dataset(dataset_name, data=coords_nwp)
                    f_extra.create_dataset(dataset_name, data=extra, compression="lzf")
                    f_y.create_dataset(dataset_name, data=y, compression="lzf")

                # After processing all times and sites, save the timestamps
                for site, times in site_times.items():
                    dataset_name = f'data_{site}'
                    f_time.create_dataset(dataset_name, data=np.array(times), compression="lzf")

process_data(sat_type="nonhrv")
