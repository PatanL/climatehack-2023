import pandas as pd
from datetime import datetime, time, timedelta, timezone
df = pd.read_parquet('/data/pv/proc.parquet')
pv_data = pd.read_parquet('/data/pv/concat.parquet').drop("generation_wh", axis=1)
import json
from tqdm import tqdm
sat_type = "nonhrv"
pv_metadata_file = "/data/pv/metadata.csv"
with open(pv_metadata_file, "r") as f:
    pv_metadata = pd.read_csv(f)
    pv_metadata.set_index("ss_id", inplace=True)
NWP_FEATURES = ["t_500", "clcl", "alb_rad", "tot_prec", "ww", "relhum_2m", "h_snow", "aswdir_s", "td_2m", "omega_1000"]
# NWP_FEATURES = ["t_500", "clct", "alb_rad", "tot_prec", "aswdifd_s"]
EXTRA_FEATURES = ["latitude_rounded", "longitude_rounded", "orientation", "tilt"]
import numpy as np
with open("./indices.json") as f:
    site_locations = {  
        data_source: {
            int(site): (int(location[0]), int(location[1]))
            for site, location in locations.items()
        }
        for data_source, locations in json.load(f).items()
    }
import h5py

def worker(df, id):
    with (
    h5py.File(f'/data/multiproc/processed_train_{id}.hdf5', 'w') as f_train,
    ):
        f_pv = f_train.create_group('pv')
        f_sat = f_train.create_group(sat_type)
        f_nwp = f_train.create_group('nwp')
        f_extra = f_train.create_group('extra')
        f_y = f_train.create_group('y')
        f_time = f_train.create_group('time')
        actually_added = 0
        for i, row in enumerate(tqdm(df.iterrows(), total=len(df), position=id)):
            if actually_added >= 45000:
                break
            if i% 1000 == 0:
                print(actually_added, i)
            try:
                ss_id, time = row[1].iloc[1], row[1].iloc[0]
                ss_id = int(ss_id)
                site = ss_id
                first_hour = slice(str(time), str(time + timedelta(minutes=55)))
                pv_features = pv_data.xs(first_hour, drop_level=False)  # type: ignore
                pv_targets = pv_data.xs(
                    slice(  # type: ignore
                        str(time + timedelta(hours=1)),
                        str(time + timedelta(hours=4, minutes=55)),
                    ),
                    drop_level=False,
                )
                # Get solar PV features and targets
                site_features = pv_features.xs(site, level=1).to_numpy().squeeze(-1)
                site_targets = pv_targets.xs(site, level=1).to_numpy().squeeze(-1)
                assert site_features.shape == (12,) and site_targets.shape == (48,)

                # Get a 128x128 crop centred on the site over the previous hour
                x, y = site_locations[sat_type][site]
                sat = (x, y)
                # nwp features
                x_nwp, y_nwp = site_locations["weather"][site]
                nwp = (x_nwp, y_nwp)
                            
                # extra features
                extra = pv_metadata.loc[site, EXTRA_FEATURES].to_numpy().astype(np.float32)
                assert extra.shape == (len(EXTRA_FEATURES),)
                data = (site_features, sat, nwp, extra, site_targets, time)
                timen = data[5].replace(tzinfo=timezone.utc) 
                f_pv.create_dataset(f'data_{actually_added}', data=data[0], compression="lzf")
                f_sat.create_dataset(f'data_{actually_added}', data=data[1], compression="lzf")
                f_nwp.create_dataset(f'data_{actually_added}', data=data[2], compression="lzf")
                f_extra.create_dataset(f'data_{actually_added}', data=data[3], compression="lzf")
                f_y.create_dataset(f'data_{actually_added}', data=data[4], compression="lzf")
                f_time.create_dataset(f'data_{actually_added}', data=np.array([timen.timestamp()]), compression="lzf")
                actually_added += 1
            except AssertionError:
                continue
            except Exception as e:
                continue

import pandas as pd
from joblib import Parallel, delayed

# create as many processes as there are CPUs on your machine
num_proc = 8
# calculate the chunk size as an integer
chunk_size = int(df.shape[0]/num_proc)

df.reset_index(inplace=True)
# this solution was reworked from the above link.
# will work even if the length of the dataframe is not evenly divisible by num_processes
chunks = [df.iloc[df.index[i:i + chunk_size]] for i in range(0, len(df), chunk_size)]
Parallel(n_jobs=num_proc)(delayed(worker)(chunk, i) for i, chunk in enumerate(chunks))