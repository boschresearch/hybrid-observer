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
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader

from hybrid_observer.configs.global_configuration import GlobalConfig
from hybrid_observer.training.folder_management.folder_factory import FolderFactory
from hybrid_observer.training.trainer_factory import TrainerFactory
from hybrid_observer.configs.config_container import get_configs
from hybrid_observer.utils import (
    set_random_seeds,
    append_rollouts_with_rs,
    compute_acc_error_over_time,
    plot_errors,
    get_metrics,
    get_mean_and_std,
    safe_data,
)
import numpy as np
import os
import timeit


def plot_training_status(X: torch.Tensor, solRef: torch.Tensor, simRef: torch.Tensor, sim: torch.Tensor):
    plt.plot(X[0, :, :], label="training signal")
    plt.plot(solRef, label="ground truth")
    plt.plot(simRef, label="simulator")
    plt.plot(sim[0, :, :], label="learned simulator signal")
    plt.legend()
    plt.show()


def save_runtimes(parentfolder: Path, times_array: np.ndarray):
    with open(os.path.join(*[parentfolder, "runtime_mean.txt"]), "w") as f:
        f.write(str(np.mean(times_array)))
    with open(os.path.join(*[parentfolder, "runtime_std.txt"]), "w") as f:
        f.write(str(np.std(times_array)))


def normalize(sol_ref):
    num_data = sol_ref.shape[0]
    sol_ref = sol_ref[:num_data]
    mean = sum(sol_ref) / num_data
    std = np.sqrt(sum((sol_ref - mean) ** 2) / (num_data - 1))
    sol_ref = (sol_ref - mean) / std
    return sol_ref


 
