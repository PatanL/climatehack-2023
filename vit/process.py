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
from tqdm import tqdm
import h5py
import hdf5plugin
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
            if current_time and np.random.rand() < 0.8:
                yield current_time
                
            minutes_to_add = int(np.random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]))
            current_time += timedelta(minutes=minutes_to_add, hours=1)
            
        date += timedelta(days=1)

def worker(dates, sat_type):  
    with open("./indices.json") as f:
        site_locations = {  
            data_source: {
                int(site): (int(location[0]), int(location[1]))
                for site, location in locations.items()
            }
            for data_source, locations in json.load(f).items()
        }
    sites = list(site_locations[sat_type].keys())
    
    pv_metadata_file = "/data/pv/metadata.csv"
    with open(pv_metadata_file, "r") as f:
        pv_metadata = pd.read_csv(f)
        pv_metadata.set_index("ss_id", inplace=True)
      
    i_train, i_val = -1, -1
    for year, month in dates:       
        pv_file_path = f"/data/pv/{year}/{month}.parquet"
        sat_file_path = f"/data/satellite-{sat_type}/{year}/{month}.zarr.zip"
        nwp_file_path = f"/data/weather/{year}/{month}.zarr.zip"
        
        pv_data = pd.read_parquet(pv_file_path).drop("generation_wh", axis=1)
        sat_data = xr.open_dataset(sat_file_path, engine="zarr", chunks="auto", consolidated=True)
        nwp_data = xr.open_dataset(nwp_file_path, engine="zarr", chunks="auto", consolidated=True)
        
        for time in get_image_times(year, month):
            first_hour = slice(str(time), str(time + timedelta(minutes=55)))

            pv_features = pv_data.xs(first_hour, drop_level=False)  # type: ignore
            pv_targets = pv_data.xs(
                slice(  # type: ignore
                    str(time + timedelta(hours=1)),
                    str(time + timedelta(hours=4, minutes=55)),
                ),
                drop_level=False,
            )

            sat_features = sat_data["data"].sel(time=first_hour).to_numpy()
            if sat_features.shape[0] != 12 or np.isnan(sat_features).any():
                continue

            for site in sites:
                if np.random.rand() < 0.7:
                    continue
                try:
                    # Get solar PV features and targets
                    site_features = pv_features.xs(site, level=1).to_numpy().squeeze(-1)
                    site_targets = pv_targets.xs(site, level=1).to_numpy().squeeze(-1)
                    assert site_features.shape == (12,) and site_targets.shape == (48,)

                    # Get a 128x128 crop centred on the site over the previous hour
                    x, y = site_locations[sat_type][site]
                    if sat_type == "hrv":
                        sat = sat_features[:, y - 64 : y + 64, x - 64 : x + 64, 0]                        
                        assert sat.shape == (12, 128, 128)
                    else:
                        sat = sat_features[:, y - 64 : y + 64, x - 64 : x + 64, :]
                        sat = sat.transpose(3, 0, 1, 2)
                        assert sat.shape == (11, 12, 128, 128)

                    # nwp features
                    nan_nwp = False
                    x_nwp, y_nwp = site_locations["weather"][site]
                    
                    # NWP HOURS: [T - 1h, T, T + 1h, T + 2h, T + 3h, T + 4h]. Granularity: 1hr
                    # where T is the forecast start time, NOT the past start time
                    T = time + timedelta(hours=1)
                    
                    # Check if time is on the hour or not
                    if T.minute == 0:
                        nwp_hours = slice(str(T - timedelta(hours=1)), str(T + timedelta(hours=4)))
                    else:
                        nwp_hours = slice(str(T - timedelta(hours=1, minutes=time.minute)), str(T + timedelta(hours=4) - timedelta(minutes=time.minute)))

                    nwp_features_arr = []
                    for feature in NWP_FEATURES:
                        data = nwp_data[feature].sel(time=nwp_hours).to_numpy()
                                    
                        if data.shape[0] != 6 or np.isnan(data).any():
                            nan_nwp = True
                            break

                        # 128x128 crop
                        data = data[:, y_nwp - 64 : y_nwp + 64, x_nwp - 64 : x_nwp + 64]
                        assert data.shape == (6, 128, 128)
                                    
                        # Normalize data
                        # data = (data - normalization_values["nwp"][feature]["min"]) / (normalization_values["nwp"][feature]["max"] - normalization_values["nwp"][feature]["min"])

                        nwp_features_arr.append(data)

                    if nan_nwp:
                        continue

                    nwp = np.stack(nwp_features_arr, axis=0)
                                
                    # extra features
                    extra = pv_metadata.loc[site, EXTRA_FEATURES].to_numpy().astype(np.float32)
                    assert extra.shape == (len(EXTRA_FEATURES),)
                        
                    # 80 20 split
                    # if np.random.rand() < 0.2:
                    #     # Get the next dataset index 
                    #     set_type = "val"
                    #     i_val += 1
                    #     yield i_val, set_type, (site_features, sat, nwp, extra, site_targets)
                    # else:
                    #     set_type = "train"
                    #     i_train += 1
                    #     yield i_train, set_type, (site_features, sat, nwp, extra, site_targets)
                
                    # Validation on 2021, training on 2020
                    if year == 2021:
                        # Get the next dataset index 
                        set_type = "train"
                        i_val += 1
                        yield i_val, set_type, (site_features, sat, nwp, extra, site_targets, first_hour)
                    else:
                        set_type = "val"
                        i_train += 1
                        yield i_train, set_type, (site_features, sat, nwp, extra, site_targets, first_hour)
                
                except:
                    # print(e)
                    continue
           

def process_data(sat_type):
    with (
        h5py.File(f'/data/processed_data/processed_train_{os.environ["START_MONTH"]}.hdf5', 'w') as f_train,
        h5py.File(f'/data/processed_data/processed_val_{os.environ["START_MONTH"]}.hdf5', 'w') as f_val,
        ):
            f_pv = f_train.create_group('pv')
            f_sat = f_train.create_group(sat_type)
            f_nwp = f_train.create_group('nwp')
            f_extra = f_train.create_group('extra')
            f_y = f_train.create_group('y')
            f_time = f_train.create_group('first_hour')

        
            f_pv_val = f_val.create_group('pv')
            f_sat_val = f_val.create_group(sat_type)
            f_nwp_val = f_val.create_group('nwp')
            f_extra_val = f_val.create_group('extra')
            f_y_val = f_val.create_group('y')
            f_time_val = f_val.create_group('first_hour')

        
            for i, set_type, data in tqdm(worker([(year, month) for year in range(2021, 2022) for month in range(int(os.environ['START_MONTH']), int(os.environ['END_MONTH']) + 1)], sat_type)):
                # (pv, sat, nwp, extra, y) = data
                if set_type == "train":
                    f_pv.create_dataset(f'data_{i}', data=data[0], compression="lzf")
                    f_sat.create_dataset(f'data_{i}', data=data[1], compression="lzf")
                    f_nwp.create_dataset(f'data_{i}', data=data[2], compression="lzf")
                    f_extra.create_dataset(f'data_{i}', data=data[3], compression="lzf")
                    f_y.create_dataset(f'data_{i}', data=data[4])
                    f_time.create_dataset(f'data_{i}', data=data[5])
                else:                    
                    f_pv_val.create_dataset(f'data_{i}', data=data[0], compression="lzf")
                    f_sat_val.create_dataset(f'data_{i}', data=data[1], compression="lzf")
                    f_nwp_val.create_dataset(f'data_{i}', data=data[2], compression="lzf")
                    f_extra_val.create_dataset(f'data_{i}', data=data[3], compression="lzf")
                    f_y_val.create_dataset(f'data_{i}', data=data[4], compression="lzf")
                    f_y_val.create_dataset(f'data_{i}', data=data[5], compression="lzf")


NWP_FEATURES = ["t_500", "clcl", "alb_rad", "tot_prec", "ww", "relhum_2m", "h_snow", "aswdir_s", "td_2m", "omega_1000"]
# NWP_FEATURES = ["t_500", "clct", "alb_rad", "tot_prec", "aswdifd_s"]
EXTRA_FEATURES = ["latitude_rounded", "longitude_rounded", "orientation", "tilt"]
process_data(sat_type="nonhrv")