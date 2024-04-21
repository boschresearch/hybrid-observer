# Copyright (c) 2024 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Author: Katharina Ensinger (katharina.ensinger@bosch.com)

import torch
from pathlib import Path
from typing import Optional, Union
from hybrid_observer.model.available_models import TimeDependentTSModel
import matplotlib.pyplot as plt
import os
from torch.optim import Optimizer

from hybrid_observer.training.optimizer.optimizers import OptimizerFactory


def RMSE(predictions: torch.Tensor, reference: torch.Tensor):
    RMSE = torch.sqrt(torch.mean((predictions - reference) ** 2))
    return RMSE


class Evaluate(torch.nn.Module):
    """
    training module, provided with model, loss, optimizer, training_steps,
    callback options, dataloader and device, it returns a trained model
    Args:
       model (torch.nn.Module): model.
       opt (torch.optim): optimizer.
    Returns:
       model (torch.nn.Module): model.
       opt (torch.optim): optimizer..

    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        opt: Optional[Optimizer] = None,
        postprocessor: Optional = None,
    ):
        super(Evaluate, self).__init__()
        self.model = model if model is not None else TimeDependentTSModel()
        self.opt = opt if opt is not None else OptimizerFactory(model=self.model).build()
        self.postprocessor = postprocessor

    def get_model(self, path: Path):
        """
        loads a model from the path and returns model, epoch and loss
        """
        self.model.load_state_dict(torch.load(path)["model_state_dict"])
        # optimizer = self.opt.load_state_dict(torch.load(path))
        epoch = torch.load(path)["epoch"]
        loss = torch.load(path)["loss"]
        return self.model, epoch, loss

    @staticmethod
    def rollout_model(model, data: torch.Tensor):
        """
        produces rollout from loaded model
        """
        # model.eval()
        with torch.no_grad():
            output = model.rollout_wrapper(data)
        return output

    @staticmethod
    def plot_evaluation(test_X: list, output: list):
        plt.plot(test_X[0][0, :, :], label="ground truth")
        plt.plot(test_X[2][0, :, :], label="simulator")
        if output[0] is not None:
            plt.plot((output[0][0, :, :]).detach().numpy(), label="learned simulator signal")
        if output[1] is not None:
            plt.plot((output[1][0, :, :]).detach().numpy(), label="learning-based signal")
        if output[2] is not None:
            plt.plot((output[2][0, :, :]).detach().numpy(), label="full signal")
        if len(output) > 3:
            if output[3] is not None:
                plt.plot((output[3][0, :, :]).detach().numpy(), label="simulator")
        plt.legend()
        plt.show()
        plt.close()

    @staticmethod
    def compute_RMSE(test_X: list, output: list):
        """
        compute RMSE
        """
        return torch.sqrt(torch.mean((output[2][0, :, :] - test_X[0][0, :, :]) ** 2))
