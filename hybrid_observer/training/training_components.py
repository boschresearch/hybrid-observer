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

from os import getenv

import torch
import numpy as np
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from typing import Optional, List
from dotenv import find_dotenv, load_dotenv
import matplotlib.pyplot as plt
from hybrid_observer.configs.training.training_config import AdamConfig
from hybrid_observer.model.available_models import TimeDependentTSModel
from hybrid_observer.training.loss.losses import RegularizedDataSimLoss, StandardLoss
from typing import Tuple, Union
from hybrid_observer.training.callbacks.available_callbacks import Callbacks, PlotDataAndSim
from pathlib import Path
from hybrid_observer.model.interfaces import Rollouts, TSModel
from hybrid_observer.training.optimizer.optimizers import OptimizerFactory
from hybrid_observer.training.preprocessing import Preprocessor, Postprocessor

DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
Parameter = torch.nn.Parameter


def remove_data(sim: np.ndarray, missing_samples: int):
    """
    remove data from array in order to simulate a simulator with buggy behavior.
    """
    if missing_samples is not None:
        N = sim.shape[0]
        indices = np.random.choice(N, missing_samples, replace=False, p=None)
        sim[indices] = "nan"
    return sim


class TrainTest:
    """
    class for loading the data depending on the type, either training data, training data and simulator or ... + time
    is loaded
    Args:
        training_split (int): index, until which data correspond to training data.
        validation_int(int): index, from which on trajectory data are considered for validation.


    Returns:
        training_split (int): index, until which data correspond to training data.
        validtaion_int(int): index, from which on trajectory data are considered for validation.

    """

    def __init__(
        self,
        training_split: int = 500,
        validation_ind: int = 0,
        remove_n_train_data: Optional[int] = None,
        remove_n_eval_data: Optional[int] = None,
        n_dimensional: Optional[bool] = False,
        preprocessor: Optional[Preprocessor] = None,
    ):
        super(TrainTest, self).__init__()
        self.training_split = training_split
        self.validation_ind = validation_ind
        self.remove_n_train_data = remove_n_train_data
        self.remove_n_eval_data = remove_n_eval_data
        self.n_dimensional = n_dimensional
        self.preprocessor = preprocessor

    def get_right_dimensionality(self, data: torch.Tensor):
        if self.n_dimensional:
            return data
        else:
            return data.unsqueeze(1)

    def load_data(self, path: Union[Path, str]) -> Tuple[list[list], list]:
        """

        Returns:
            training_data List[(torch.Tensor)]: training_data
            validation_data (torch.Tensor): validation_data

        """
        npzfile = np.load(path)
        sorted(npzfile.files)
        sol = npzfile["sol"]
        sol_ref = npzfile["sol_ref"]
        if self.preprocessor is not None:
            training_data = [
                [self.get_right_dimensionality(self.preprocessor.preprocess(torch.Tensor(sol[: self.training_split])))]
            ]
        else:
            training_data = [[self.get_right_dimensionality(torch.Tensor(sol[: self.training_split]))]]
        if self.preprocessor is not None:
            validation_data = [
                self.get_right_dimensionality(
                    self.preprocessor.preprocess(torch.Tensor(sol_ref[self.validation_ind :]))
                ).unsqueeze(0),
                self.get_right_dimensionality(
                    self.preprocessor.preprocess(torch.Tensor(sol[self.validation_ind : self.training_split]))
                ).unsqueeze(0),
            ]
        else:
            validation_data = [
                self.get_right_dimensionality(torch.Tensor(sol_ref[self.validation_ind :])).unsqueeze(0),
                self.get_right_dimensionality(torch.Tensor(sol[self.validation_ind :])).unsqueeze(0),
            ]
        return training_data, validation_data

    def load_data_with_sim(self, path: Union[Path, str]) -> Tuple[list[list], list]:
        """

        Returns:
            training_data List[(torch.Tensor), (torch.Tensor)]: training_data, simulator_data
            validation_data (torch.Tensor): validation_data

        """
        path = path if isinstance(path, Path) else Path(path)
        if not path.exists():
            raise ValueError(
                f"we are missing the file {path}. Check the datasets folder, you might need to run one of the "
                f"generate... scripts."
            )
        npzfile = np.load(path)
        sorted(npzfile.files)
        sol = npzfile["sol"]
        sol_ref = npzfile["sol_ref"]
        sim = npzfile["sim"]
        training_data = [
            [self.get_right_dimensionality(torch.Tensor((sol[: self.training_split])))],
            [
                self.get_right_dimensionality(
                    torch.Tensor(remove_data(sim[: self.training_split], self.remove_n_train_data))
                )
            ],
        ]
        validation_data = [
            self.get_right_dimensionality(torch.Tensor(sol_ref[self.validation_ind :])).unsqueeze(0),
            self.get_right_dimensionality(torch.Tensor(sol[self.validation_ind :])).unsqueeze(0),
            self.get_right_dimensionality(
                torch.Tensor(remove_data(sim[self.validation_ind :], self.remove_n_eval_data))
            ).unsqueeze(0),
        ]

        return training_data, validation_data

    def load_data_with_sim_and_time(self, path: Union[Path, str]) -> Tuple[list[list], list]:
        """

        Returns:
            training_data List[(torch.Tensor), (torch.Tensor), torch.Tensor]: training_data, simulator_data, time
            validation_data (torch.Tensor): validation_data

        """
        npzfile = np.load(path)
        sorted(npzfile.files)
        sol = npzfile["sol"]
        sol_ref = npzfile["sol_ref"]
        sim = npzfile["sim"]
        t = npzfile["t"]
        training_data = [
            [self.get_right_dimensionality(torch.Tensor(sol[: self.training_split]))],
            [self.get_right_dimensionality(torch.Tensor(sim[: self.training_split]))],
            [torch.Tensor(t[: self.training_split]).unsqueeze(1)],
        ]
        validation_data = [
            self.get_right_dimensionality(torch.Tensor(sol_ref[self.validation_ind :])).unsqueeze(0),
            self.get_right_dimensionality(torch.Tensor(sol[self.validation_ind :])).unsqueeze(0),
            self.get_right_dimensionality(torch.Tensor(sim[self.validation_ind :])).unsqueeze(0),
            torch.Tensor(t[self.validation_ind :]).unsqueeze(1).unsqueeze(0),
        ]
        return training_data, validation_data


class Dataloader:
    """
    dataloader class, computing a pytorch dataloader that consists of subtrajectories of the list of training trajectories
    Args:
       batchsize (int): batchsize.
       isshuffle (bool): indicator, whether data are shuffled.
       length (int): length of subtrajectories.
       device (torch.device): device.

    Returns:
       batchsize (int): batchsize.
       isshuffle (bool): indicator, whether data are shuffled.
       length (int): length of subtrajectories.
       device (torch.device): device.

    """

    def __init__(self, batchsize: int = 50, isshuffle: bool = True, length: int = 100, device: Optional[str] = "cpu"):
        super(Dataloader, self).__init__()
        self.batchsize = batchsize
        self.isshuffle = isshuffle
        self.length = length
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess_list(self, dataset) -> list:
        """

        Args:
           dataset (list[list[torch.Tensor],list[torch.Tensor,...]:
           List of arbitrary size, List entries are lists correspding to different channels (data, simulator...)
           List[List] corresponds to list of trajectories that are used for training

        Returns:
           subtrajectory_list (list[torch.Tensors,...]: list of arbitrary size of Tensors of subtrajectories for batching

        """
        subtrajectory_list = []
        for datatype in dataset:
            for trajectory in datatype:
                data = []
                # evaluate trajectory
                trajectorySize = trajectory.shape[0]
                # indices of initial values, goes in steps of length
                arg = np.arange(0, trajectorySize - self.length)
                for j in range(arg.shape[0]):
                    # evaluate subtrajectory
                    X = FloatTensor(trajectory[arg[j] : arg[j] + self.length + 1, :]).unsqueeze(0)
                    data.append(X)
                data = torch.cat(data, 0)
                data = data.to(self.device)
                subtrajectory_list.append(data)
        return subtrajectory_list

    def dataset_from_list(self, *dataset: list) -> DataLoader:
        """

        Args:
          *dataset (list[torch.Tensors]): inputs

        Returns:
          dl (torch.utils.data.DataLoader): dataloader

        """
        dataset = torch.utils.data.TensorDataset(*dataset)
        dl = DataLoader(dataset, batch_size=self.batchsize, shuffle=self.isshuffle)
        return dl

    def get_dataloader(self, dataset) -> DataLoader:
        """

        Args:
           dataset (list[list[torch.Tensor],list[torch.Tensor,...]:
           List of arbitrary size, List entries are lists corresponding to different channels (data, simulator...)
           List[List] corresponds to list of trajectories that are used for training
        Returns:
            dl (torch.utils.data.DataLoader): dataloader

        """
        dataset = self.preprocess_list(dataset)
        dataloader = self.dataset_from_list(*dataset)
        return dataloader


class Trainer:
    """
    training module, provided with model, loss, optimizer, training_steps,
    callback options, dataloader and device, it returns a trained model
    Args:
       model (torch.nn.Module): model.
       loss (torch.nn.Module): loss function.
       opt (torch.optim): length of subtrajectories.
       training_steps (int): training epochs.
       save_epoch (int): each save_epochs, callbacks are applied (model is saved etc.).
       callbacks (Optional[list]: list of callbacks).
       dataloader (DataLoader): torch dataloader.
       device (torch.device): device.
       scheduler (Scheduler): possibility to add learning rate scheduler.
       postprocessor (Postprocessor): possiblity to postprocess data after training.

    Returns:
       model (torch.nn.Module): model.
       loss (torch.nn.Module): loss function.
       opt (torch.optim): length of subtrajectories.
       training_steps (int): training epochs.
       save_epoch (int): each save_epochs, callbacks are applied (model is saved etc.).
       callbacks (Optional[list]: list of callbacks).
       dataloader (DataLoader): torch dataloader.
       device (torch.device): device.
       scheduler (Scheduler): possibility to add learning rate scheduler.
       postprocessor (Postprocessor): possiblity to postprocess data after training.

    """

    def __init__(
        self,
        model: TSModel,
        loss: torch.nn.Module = None,
        opt: Optimizer = None,
        training_steps: int = 1000,
        save_epoch: int = 50,
        callbacks: Optional[List[Callbacks]] = None,
        device: torch.device = "cpu",
        scheduler: lr_scheduler = None,
        postprocessor: Optional[Postprocessor] = None,
    ):
        self.model: TSModel = model
        self.loss = loss if loss is not None else RegularizedDataSimLoss()
        self.opt: Optimizer = opt if opt is not None else OptimizerFactory.build(AdamConfig(), model)
        self.training_steps = training_steps
        self.save_epoch = save_epoch
        self.callbacks: List[Callbacks] = callbacks if callbacks is not None else []
        self.device = device
        self.scheduler = scheduler
        self.postprocessor: Postprocessor = postprocessor

    def train(self, test_X: list, dataloader: torch.utils.data.dataloader, random_seed: int = 50) -> torch.nn.Module:
        """

        Args:
            test_X ((n_data, trajectory_length, dim)-torch.Tensor): test data


        Returns:
            model (torch.nn.Module): training model
        """
        losses = []
        for i in range(self.training_steps):
            epoch_loss = 0
            with torch.no_grad():
                output = self.model.rollout_wrapper(test_X)
                for callback in self.callbacks:
                    callback.compute_callback(test_X, output, self.model, self.loss, i, random_seed, self.opt, losses)
            for batches in dataloader:
                """loading training data"""
                self.opt.zero_grad()
                computed_loss = self.model.fit(batches, self.loss)
                computed_loss.backward()
                self.opt.step()
                epoch_loss += computed_loss.detach()
            losses.append(epoch_loss)
            if self.scheduler is not None:
                self.scheduler.step()
        return self.model

    def compute_test_rollout(self, test_X: torch.nn.Module):
        """
        For an input trajectory test_X, compute a rollout.
        """
        with torch.no_grad():
            output = self.model.rollout_wrapper(test_X)
        return output

    def postprocess(self, output: torch.nn.Module):
        """
        For a trajectory, postprocess the trajectory.
        """
        return self.postprocessor.postprocess(output) if self.postprocessor is not None else output
