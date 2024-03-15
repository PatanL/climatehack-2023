import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))


import h5py
import torch
from competition import BaseEvaluator
from model import OurResnet2 as Model
import datetime
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class Evaluator(BaseEvaluator):
    def setup(self, model = None) -> None:
        """Sets up anything required for evaluation, e.g. loading a model."""
        if not model:
            self.model = Model().to(device)
            self.model.load_state_dict(torch.load("/data/SplitTemporalResnet2+1Combo-Full-Weather-NewOptim-ResFCNet2-ep1.pt", map_location=device))
        else:
            self.model = model
        self.model.eval()

    def predict(self, features: h5py.File):
        """Makes solar PV predictions for a test set.

        You will have to modify this method in order to use additional test set data variables
        with your model.

        Args:
            features (h5py.File): Solar PV, satellite imagery, weather forecast and air quality forecast features.

        Yields:
            Generator[np.ndarray, Any, None]: A batch of predictions.
        """
        with torch.inference_mode():
            # Select the variables you wish to use here!
            for data in self.batch(features, variables=["t_500", "clcl", "alb_rad", "tot_prec", "ww", "relhum_2m", "h_snow", "aswdir_s", "td_2m", "omega_1000", "pv", "hrv", "time"], batch_size=32):
                # Produce solar PV predictions for this batch
                pv, hrv, times = data[-3:]
                hrv = torch.unsqueeze(torch.from_numpy(hrv), 1)
                nwp = torch.from_numpy(np.stack(data[:-3])).permute(1, 0, 2, 3, 4)
                nwp = nwp.flatten(1, 2)
                # print(nwp.shape, hrv.shape)
                with torch.autocast(device_type=device):
                    yield self.model(
                        torch.from_numpy(pv).to(device, dtype=torch.float),
                        hrv.to(device, dtype=torch.float),
                        nwp.to(device,dtype=torch.float),
                    ).cpu()


if __name__ == "__main__":
    Evaluator(None).evaluate()
