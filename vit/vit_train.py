# %%
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from ocf_blosc2 import Blosc2
from torch.utils.data import DataLoader, IterableDataset
from torchinfo import summary
import json
plt.rcParams["figure.figsize"] = (20, 12)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %% [markdown]
# ## Loading the data

# %%
pv = pd.read_parquet("data/pv/2020/1.parquet").drop("generation_wh", axis=1)
for i in range(2, 13):
    pv2 = pd.read_parquet(f"data/pv/2020/{i}.parquet").drop("generation_wh", axis=1)
    pv = pd.concat([pv, pv2], axis=0)

# %%
hrv = xr.open_dataset(
    "data/satellite-hrv/2020/1.zarr.zip", engine="zarr", chunks="auto"
)
for i in range(2, 13):
    hrv2 = xr.open_dataset(
        f"data/satellite-hrv/2020/{i}.zarr.zip", engine="zarr", chunks="auto"
    )
    hrv = xr.concat((hrv, hrv2), dim="time")

# %% [markdown]
# As part of the challenge, you can make use of satellite imagery, numerical weather prediction and air quality forecast data in a `[128, 128]` region centred on each solar PV site. In order to help you out, we have pre-computed the indices corresponding to each solar PV site and included them in `indices.json`, which we can load directly. For more information, take a look at the [challenge page](https://doxaai.com/competition/climatehackai-2023).
# 

# %%
with open("indices.json") as f:
    site_locations = {
        data_source: {
            int(site): (int(location[0]), int(location[1]))
            for site, location in locations.items()
        }
        for data_source, locations in json.load(f).items()
    }

# %% [markdown]
# ### Defining a PyTorch Dataset
# 
# To get started, we will define a simple `IterableDataset` that shows how to slice into the PV and HRV data using `pandas` and `xarray`, respectively. You will have to modify this if you wish to incorporate non-HRV data, weather forecasts and air quality forecasts into your training regimen. If you have any questions, feel free to ask on the [ClimateHack.AI Community Discord server](https://discord.gg/HTTQ8AFjJp)!
# 
# **Note**: `site_locations` contains indices for the non-HRV, weather forecast and air quality forecast data as well as for the HRV data!
# 
# There are many more advanced strategies you could implement to load data in training, particularly if you want to pre-prepare training batches in advance or use multiple workers to improve data loading times.

# %%
class ChallengeDataset(IterableDataset):
    def __init__(self, pv, hrv, site_locations, sites=None, transform=None, min_date=None, max_date=None):
        self.pv = pv
        self.hrv = hrv
        self._site_locations = site_locations
        self.transform=transform
        self._sites = sites if sites else list(site_locations["hrv"].keys())
        assert (min_date and max_date), "Did not provide a min and/or max date range."
        self.min_date = min_date
        self.max_date = max_date
    def __len__(self):
        i = 0
        for _ in self._get_image_times():
            i += 1
        i *= len(self._sites)
        return i
    def _get_image_times(self):
        min_date = self.min_date
        max_date = self.max_date

        start_time = time(8)
        end_time = time(17)
        date = min_date
        while date <= max_date:
            current_time = datetime.combine(date, start_time)
            while current_time.time() < end_time:
                if current_time:
                    yield current_time

                current_time += timedelta(minutes=60)

            date += timedelta(days=1)

    def __iter__(self):
        for time in self._get_image_times():
            first_hour = slice(str(time), str(time + timedelta(minutes=55)))

            pv_features = pv.xs(first_hour, drop_level=False)  # type: ignore
            pv_targets = pv.xs(
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

                    # How might you adapt this for the non-HRV, weather and aerosol data?
                except:
                    continue
                if self.transform:
                    hrv_features = self.transform(hrv_features)
                yield site_features, hrv_features, site_targets

# %% [markdown]
# ## Train a model

# %%
BATCH_SIZE = 32
train_dataset = ChallengeDataset(pv, hrv, site_locations=site_locations, min_date=datetime(2020, 1, 1), max_date=datetime(2020, 12, 31))
dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True)
print(f"train dataset len: {len(train_dataset)}")

# %%
from submission.model import OurTransformer
model = OurTransformer(image_size=128).to(device)
criterion = nn.L1Loss()
optimiser = optim.Adam(model.parameters(), lr=1e-3)
summary(model, input_size=[(1, 12), (1, 12, 128, 128)])

# %%
EPOCHS = 10
MODEL_KEY="ViT-Tiny-Full"
from tqdm import tqdm
for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    count = 0
    for i, (pv_features, hrv_features, pv_targets) in (pbar := tqdm(enumerate(dataloader), total=len(dataloader))):
        optimiser.zero_grad()

        predictions = model(
            pv_features.to(device, dtype=torch.float),
            hrv_features.to(device, dtype=torch.float),
        )
        loss = criterion(predictions, pv_targets.to(device, dtype=torch.float))
        loss.backward()

        optimiser.step()

        size = int(pv_targets.size(0))
        running_loss += float(loss) * size
        count += size

        if i % 100 == 99:
            pbar.set_description(f"Epoch {epoch + 1}, {i + 1}: {running_loss / count}")
        if i == int(len(dataloader) * 0.5):
            print("Saving halfway-point model...")
            torch.save(model.state_dict(), f"submission/{MODEL_KEY}-ep{epoch + 1}-half.pt")

    print(f"Epoch {epoch + 1}: {running_loss / count}")
    torch.save(model.state_dict(), f"submission/{MODEL_KEY}-ep{epoch + 1}.pt")
    print("Saved model!")

# %%
# Save your model
torch.save(model.state_dict(), "submission/model.pt")

# %%



