#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import glob
plt.rcParams["figure.figsize"] = (20, 12)
# %load_ext autoreload
# %autoreload 2


# In[ ]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# In[ ]:


from dataset import ChallengeDataset
import json
with open("indices.json") as f:
    site_locations = {
        data_source: {
            int(site): (int(location[0]), int(location[1]))
            for site, location in locations.items()
        }
        for data_source, locations in json.load(f).items()
    }
dataset = ChallengeDataset('/data/pv/2021/', '/data/satellite-nonhrv_proc/2021/', "/data/weather_proc/2021/", site_locations, None)
data_loader = DataLoader(dataset, batch_size=16, pin_memory=True, num_workers=6, shuffle=False)
print(f"train dataset len: {len(dataset)}")


# In[ ]:


EPOCHS = 15
START_EPOCH = 0
LR = 1e-3
from submission.model import OurResnet2
model = OurResnet2(image_size=128, device=device).to(device)
criterion = nn.L1Loss()
optimiser = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.02)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=10, eta_min=7e-5)
summary(model, input_size=[(1, 12), (1, 11, 12, 128, 128), (1, 10, 6, 128, 128), (1, 3)])
# x = torch.randn((1, 12)).to(device)
# y = torch.randn((1, 1, 12, 128, 128)).to(device)
# z = torch.randn((1, 10, 6, 128, 128)).to(device)
# a = torch.randn((1, 3)).to(device)
# model(x, y, z, a)


# In[ ]:


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
def hasNan(tensor):
    return torch.isnan(tensor).any()


# In[9]:


MODEL_KEY="ExtraEmbedding_TemporalResnet2+1Combo-PVResFCNet2"
print(f"Training model key {MODEL_KEY}")
from tqdm import tqdm
for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    i = 0
    count = 0
    for (pv_features, hrv_features, nwp, extra, pv_targets) in (pbar := tqdm(data_loader, total=len(data_loader), ascii=True)):
        optimiser.zero_grad()
        with torch.autocast(device_type=device):
            real_extra = extra[:, 2:]
            if hasNan(pv_features) or hasNan(hrv_features) or hasNan(nwp) or hasNan(extra) or hasNan(pv_targets):
                print(f"Found nan {i}")
                continue
            predictions = model(
                pv_features.to(device,dtype=torch.float),
                hrv_features.to(device,dtype=torch.float),
                nwp.to(device,dtype=torch.float),
                real_extra.to(device,dtype=torch.float),
            )
            loss = criterion(predictions, pv_targets.to(device, dtype=torch.float))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimiser.step()

        size = int(pv_targets.size(0))
        running_loss += float(loss) * size
        count += size

        if i % 10 == 9:
            writer.add_scalar(f"Loss/train_batch_level", (running_loss / count), epoch * len(data_loader) + i)
            pbar.set_description(f"Epoch {START_EPOCH + epoch + 1}, {i + 1}: {running_loss / count}")
        if i % 100 == 99:
            print(f"Epoch {START_EPOCH + epoch + 1}, {i + 1}: {running_loss / count}")
            writer.add_scalar(f"Loss/train_ep_level", (running_loss / count), START_EPOCH + epoch + 1)
        if i % 3000 == 2999:
            torch.save(model.state_dict(), f"./cpts/{MODEL_KEY}-ep{START_EPOCH + epoch + 1}.pt")
        i += 1
    lr_scheduler.step() 
    current_lr = lr_scheduler.get_last_lr()[0]
    print(f"Epoch {START_EPOCH + epoch + 1}: {running_loss / count} (LR: {current_lr})")
    writer.add_scalar(f"LR", current_lr, START_EPOCH + epoch + 1)
    torch.save(model.state_dict(), f"./cpts/{MODEL_KEY}-ep{START_EPOCH + epoch + 1}.pt")
    print("Saved model!")


# In[ ]:





# In[ ]:


for i in range(100):
    pv_features, hrv_features, weather_features, extra, pv_targets = dataset[i]
    print(hrv_features.shape, weather_features.shape)


# In[ ]:




