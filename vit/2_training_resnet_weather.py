#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
# %load_ext autoreload
# %autoreload 2


# In[2]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# ## Train a model

# In[3]:


from dataset import HDF5Dataset
dataset = HDF5Dataset("./data/processed_data/processed_train.hdf5", True, True, True, True)
data_loader = DataLoader(dataset, batch_size=32, pin_memory=True, num_workers=8, shuffle=True)
print(f"train dataset len: {len(dataset)}")


# In[12]:


from submission.model import OurResnet2
model = OurResnet2(image_size=128).to(device)
model.load_state_dict(torch.load("submission/OurResnetCombo-Full-NoWeather-ep16.pt", map_location=device))
criterion = nn.L1Loss()
optimiser = optim.Adam(model.parameters(), lr=1e-3)
summary(model, input_size=[(1, 12), (1, 12, 1, 128, 128), (1, 6, 10, 128, 128)])
# x = torch.randn((1, 12)).to(device)
# y = torch.randn((1, 12, 1, 128, 128)).to(device)
# z = torch.randn((1, 6, 10, 128, 128)).to(device)
# model(x, y, z)


# In[13]:


EPOCHS = 100
START_EPOCH = 16
MODEL_KEY="OurResnetCombo-Full-Weather"
print(f"Training model key {MODEL_KEY}")
from tqdm import tqdm
for epoch in range(EPOCHS):
    with torch.no_grad():
        from validate import main
        main(model=model)
    model.train()

    running_loss = 0.0
    count = 0
    for i, (pv_features, hrv_features, nwp, extra, pv_targets) in (pbar := tqdm(enumerate(data_loader), total=len(data_loader))):
        optimiser.zero_grad()
        with torch.autocast(device_type=device):
            nwp = nwp.permute(0, 2, 1, 3, 4)
            hrv_features = torch.unsqueeze(hrv_features, 2) # channels as first dim then number of "frames"
            predictions = model(
                pv_features.to(device,dtype=torch.float),
                hrv_features.to(device,dtype=torch.float),
                nwp.to(device,dtype=torch.float),
            )
            loss = criterion(predictions, pv_targets.to(device, dtype=torch.float))
        loss.backward()

        optimiser.step()

        size = int(pv_targets.size(0))
        running_loss += float(loss) * size
        count += size

        if i % 50 == 49:
            pbar.set_description(f"Epoch {START_EPOCH + epoch + 1}, {i + 1}: {running_loss / count}")

    print(f"Epoch {START_EPOCH + epoch + 1}: {running_loss / count}")
    torch.save(model.state_dict(), f"submission/{MODEL_KEY}-ep{START_EPOCH + epoch + 1}.pt")
    print("Saved model!")


# In[ ]:


# Save your model
# torch.save(model.state_dict(), "submission/model.pt")


# In[ ]:




