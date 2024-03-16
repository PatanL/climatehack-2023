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
plt.rcParams["figure.figsize"] = (20, 12)
# %load_ext autoreload
# %autoreload 2


# In[ ]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# In[ ]:


from dataset import HDF5Dataset
dataset = HDF5Dataset("./data/ds14_processed_data/processed_train.hdf5", True, True, True, True)
data_loader = DataLoader(dataset, batch_size=16, pin_memory=True, num_workers=8, shuffle=True)
print(f"train dataset len: {len(dataset)}")


# In[ ]:


EPOCHS = 200
START_EPOCH = 0
from submission.model import OurResnet2
model = OurResnet2(image_size=128, device=device).to(device)
criterion = nn.L1Loss()
optimiser = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.02)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=30, eta_min=3e-4)
summary(model, input_size=[(1, 12), (1, 1, 12, 128, 128), (1, 10, 6, 128, 128), (1, 3)])
# x = torch.randn((1, 12)).to(device)
# y = torch.randn((1, 1, 12, 128, 128)).to(device)
# z = torch.randn((1, 10, 6, 128, 128)).to(device)
# a = torch.randn((1, 3)).to(device)
# model(x, y, z, a)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


# In[ ]:


MODEL_KEY="ExtraEmbedding_TemporalResnet2+1Combo-DeepFC"
print(f"Training model key {MODEL_KEY}")
from tqdm import tqdm
for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    count = 0
    for i, (pv_features, hrv_features, nwp, extra, pv_targets) in (pbar := tqdm(enumerate(data_loader), total=len(data_loader), ascii=True)):
        optimiser.zero_grad()
        with torch.autocast(device_type=device):
            real_extra = extra[:, 2:]
            hrv_features = torch.unsqueeze(hrv_features, 1)
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
            writer.add_scalar(f"Loss/ep{START_EPOCH + epoch + 1}", (running_loss / count), i)
            pbar.set_description(f"Epoch {START_EPOCH + epoch + 1}, {i + 1}: {running_loss / count}")
        if i % 100 == 99:
            print(f"Epoch {START_EPOCH + epoch + 1}, {i + 1}: {running_loss / count}")

    lr_scheduler.step()
    current_lr = lr_scheduler.get_last_lr()[0]
    print(f"Epoch {START_EPOCH + epoch + 1}: {running_loss / count} (LR: {current_lr})")
    writer.add_scalar(f"Loss/train", (running_loss / count), START_EPOCH + epoch + 1)
    writer.add_scalar(f"LR", current_lr, START_EPOCH + epoch + 1)
    torch.save(model.state_dict(), f"/data/{MODEL_KEY}-ep{START_EPOCH + epoch + 1}.pt")
    print("Saved model!")

