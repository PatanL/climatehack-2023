import torch
from torch.utils.data import DataLoader, Dataset
import h5py

BATCH_SIZE = 32
class HDF5Dataset(Dataset):
    def __init__(self, file, pv, sat, nwp, extra):
        self.file = file
        self.pv = pv
        self.sat = sat
        self.nwp = nwp
        self.extra = extra
        # Open the file quickly to get the number of keys
        with h5py.File(self.file, 'r') as f:
            self.length = len(f['pv'].keys())

    def __len__(self):
        return self.length - 1

    def __getitem__(self, idx):
        # Open the file each time to ensure lazy loading
        with h5py.File(self.file, 'r') as f:
            data_name = f'data_{idx}'
            data = []
            
            if self.pv:
                data.append(torch.from_numpy(f['pv'][data_name][...]))
            if self.sat:
                data.append(torch.from_numpy(f['hrv'][data_name][...]))
            if self.nwp:
                data.append(torch.from_numpy(f['nwp'][data_name][...]))
            if self.extra:
                data.append(torch.from_numpy(f['extra'][data_name][...]))
                
            data.append(torch.from_numpy(f['y'][data_name][...]))
            return data