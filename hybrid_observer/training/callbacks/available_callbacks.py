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

import matplotlib.pyplot as plt
import torch
import os
from typing import Union, Sequence, List
import numpy as np
from pathlib import Path

from torch import Tensor
from torch.optim import Optimizer


def callback_iter(i: int, callback_epoch):
    return i % callback_epoch == 0


def callback_number(i: int, callback_epoch):
    return int(i / callback_epoch)


class Callbacks:
    """
    interface for callbacks
    """

    def compute_callback(
        self,
        test_X: List[Tensor],
        output: Sequence[Tensor],
        model: torch.nn.Module,
        loss: torch.nn.Module,
        training_ind: int,
        random_seed: int,
        opt: Optimizer,
        losses: list,
    ):
        """
        Inputs:
            test_X (List(Tensor)): list of reference tensors for test
            output Sequence(Tensor): sequency of predictions during test time
            model (torch.nn.Module): model
            loss (torch.nn.Module): loss
            training_ind (int): current training epoch to check, whether it is time for the callback
            random_seed (int): random_seed )relevant for saving results
            opt (Optimizer): optimizer
            losses (list): list of collected epoch losses
        do computations
        """
        raise NotImplementedError


class PlotDataAndSim(Callbacks):
    """
    plots training data (observations and simulator) and predictions (observations and simulator)

    Args:
        callback_epoch (int): every callback_epoch epochs, the callback is computed
        folder (Path): folder, where results are stored
        show_plot (bool): indicate. whether plot is showed or only saved
    Attributes:
        callback_epoch (int): every callback_epoch epochs, the callback is computed
        folder (Path): folder, where results are stored
        show_plot (bool): indicate. whether plot is showed or only saved
    """

    def __init__(self, callback_epoch: int = 50, folder: Path = "results", show_plot: bool = True):
        super(PlotDataAndSim, self).__init__()
        self.callback_epoch = callback_epoch
        self.folder = folder
        self.show_plot = show_plot

    def compute_callback(
        self,
        test_X: list,
        output: list,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        training_ind: int,
        random_seed: int,
        opt: Union[torch.optim.Adam, torch.optim.SGD],
        losses: list,
    ):
        """
        see interface
        """
        if callback_iter(training_ind, self.callback_epoch):
            plt.plot(test_X[0][0, :, :], label="ground truth")
            plt.plot(test_X[2][0, :, :], label="simulator")
            if output[0] is not None:
                plt.plot(output[0][0, :, :], label="learned simulator signal")
            if output[1] is not None:
                plt.plot(output[1][0, :, :], label="learning-based signal")
            if output[2] is not None:
                plt.plot(output[2][0, :, :], label="full signal")
            if len(output) > 3:
                if output[3] is not None:
                    plt.plot(output[3][0, :, :], label="simulator")
            plt.legend()
            if self.folder.exists():
                string_list = [
                    self.folder,
                    str(random_seed) + "plots" + str(callback_number(training_ind, self.callback_epoch)),
                ]
                plt.savefig(os.path.join(*string_list))
            if self.show_plot:
                plt.show()
            plt.close()


class PlotData(Callbacks):
    def __init__(self, callback_epoch: int = 50, folder: Path = "results", show_plot=True):
        super(PlotData, self).__init__()
        self.callback_epoch = callback_epoch
        self.folder = folder
        self.show_plot = show_plot

    """
    plots training data (observations) and predictions (observations)

    Args:
        callback_epoch (int): every callback_epoch epochs, the callback is computed
        folder (Path): folder, where results are stored
        show_plot (bool): indicate. whether plot is showed or only saved
    Attributes:
        callback_epoch (int): every callback_epoch epochs, the callback is computed
        folder (Path): folder, where results are stored
        show_plot (bool): indicate. whether plot is showed or only saved
    """

    def compute_callback(
        self,
        test_X: list,
        output: list,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        training_ind: int,
        random_seed: int,
        opt: Union[torch.optim.Adam, torch.optim.SGD],
        losses: list,
    ):
        """
        see interface
        """
        if callback_iter(training_ind, self.callback_epoch):
            plt.plot(output[2][0, :, :], label="training signal")
            plt.plot(test_X[0][0, :, :], label="ground truth")
            plt.legend()
            if self.folder.exists():
                string_list = [
                    self.folder,
                    str(random_seed) + "plots" + str(callback_number(training_ind, self.callback_epoch)),
                ]
                plt.savefig(os.path.join(*string_list))
            if self.show_plot:
                plt.show()
            plt.close()


class SaveRollouts(Callbacks):
    """
    save rollouts as txt

    Args:
        callback_epoch (int): every callback_epoch epochs, the callback is computed
        folder (Path): folder, where results are stored
    Attributes:
        callback_epoch (int): every callback_epoch epochs, the callback is computed
        folder (Path): folder, where results are stored
    """

    def __init__(self, callback_epoch: int = 50, folder: Path = "results"):
        super(Callbacks, self).__init__()
        self.callback_epoch = callback_epoch
        self.folder = folder

    def compute_callback(
        self,
        test_X: list,
        output: list,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        training_ind: int,
        random_seed: int,
        opt: Union[torch.optim.Adam, torch.optim.SGD],
        losses: list,
    ):
        """
        see interfafe
        """
        if callback_iter(training_ind, self.callback_epoch):
            string_list = [
                self.folder,
                str(random_seed) + "rollout" + str(callback_number(training_ind, self.callback_epoch)) + ".txt",
            ]
            string_list_sim = [
                self.folder,
                str(random_seed) + "rollout_sim" + str(callback_number(training_ind, self.callback_epoch)) + ".txt",
            ]
            string_list_learned = [
                self.folder,
                str(random_seed) + "rollout_res" + str(callback_number(training_ind, self.callback_epoch)) + ".txt",
            ]
            string_list_leaned_sim = [
                self.folder,
                str(random_seed)
                + "predicted_simulator"
                + str(callback_number(training_ind, self.callback_epoch))
                + ".txt",
            ]
            np.savetxt(os.path.join(*string_list), output[2][0, :, :].numpy())
            if output[0] is not None:
                np.savetxt(os.path.join(*string_list_leaned_sim), output[0][0, :, :].numpy())
            if output[1] is not None:
                np.savetxt(os.path.join(*string_list_sim), output[1][0, :, :].numpy())
            if output[3] is not None:
                np.savetxt(os.path.join(*string_list_learned), output[1][0, :, :].numpy())


class SaveModel(Callbacks):
    """
    save model

    Args:
        callback_epoch (int): every callback_epoch epochs, the callback is computed
        folder (Path): folder, where results are stored
    Attributes:
        callback_epoch (int): every callback_epoch epochs, the callback is computed
        folder (Path): folder, where results are stored
    """

    def __init__(self, callback_epoch: int = 50, folder: Path = "results"):
        super(SaveModel, self).__init__()
        self.callback_epoch = callback_epoch
        self.folder = folder

    def compute_callback(
        self,
        test_X: list,
        output: list,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        training_ind: int,
        random_seed: int,
        opt: Union[torch.optim.Adam, torch.optim.SGD],
        losses: list,
    ):
        """
        see interface
        """
        if callback_iter(training_ind, self.callback_epoch):
            string_list = [
                self.folder,
                str(random_seed) + "model" + str(callback_number(training_ind, self.callback_epoch)) + ".pt",
            ]
            path = os.path.join(*string_list)
            torch.save(
                {
                    "epoch": training_ind,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "loss": loss,
                },
                path,
            )


class ComputeLosses(Callbacks):
    """
    computes loss and saves them

    Args:
        callback_epoch (int): every callback_epoch epochs, the callback is computed
        folder (Path): folder, where results are stored
    Attributes:
        callback_epoch (int): every callback_epoch epochs, the callback is computed
        folder (Path): folder, where results are stored
    """

    def __init__(self, callback_epoch: int = 50, folder: Path = "results"):
        super(ComputeLosses, self).__init__()
        self.callback_epoch = callback_epoch
        self.folder = folder

    def compute_callback(
        self,
        test_X: list,
        output: list,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        training_ind: int,
        random_seed: int,
        opt: Union[torch.optim.Adam, torch.optim.SGD],
        losses: list,
    ):
        """
        see interface
        """
        if callback_iter(training_ind, self.callback_epoch):
            string_list = [
                self.folder,
                str(random_seed) + "losses" + str(callback_number(training_ind, self.callback_epoch)) + "txt",
            ]
            if losses:
                np.savetxt(os.path.join(*string_list), torch.stack(losses).numpy())


class PlotLosses(Callbacks):
    """
    plots epoch loss curve

    Args:
        callback_epoch (int): every callback_epoch epochs, the callback is computed
        folder (Path): folder, where results are stored
        show_plot (bool): indicate. whether plot is showed or only saved
    Attributes:
        callback_epoch (int): every callback_epoch epochs, the callback is computed
        folder (Path): folder, where results are stored
        show_plot (bool): indicate. whether plot is showed or only saved
    """

    def __init__(self, callback_epoch: int = 50, folder: Path = "results", show_plot=True):
        super(PlotLosses, self).__init__()
        self.callback_epoch = callback_epoch
        self.folder = folder
        self.show_plot = show_plot

    def compute_callback(
        self,
        test_X: list,
        output: list,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        training_ind: int,
        random_seed: int,
        opt: Union[torch.optim.Adam, torch.optim.SGD],
        losses: list,
    ):
        """
        see interface
        """
        if callback_iter(training_ind, self.callback_epoch):
            if losses:
                plt.plot(torch.stack(losses).numpy())
                if self.folder.exists():
                    string_list = [
                        self.folder,
                        str(random_seed) + "loss" + str(callback_number(training_ind, self.callback_epoch)),
                    ]
                    plt.savefig(os.path.join(*string_list))
                if self.show_plot:
                    plt.show()
                plt.close()
