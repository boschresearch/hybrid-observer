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
from torch.optim import Optimizer, lr_scheduler

from hybrid_observer.configs.training.training_config import AdamConfig, SGDConfig, ImplementedSchedulers, StepLRConfig
import itertools
from typing import Union


class OptimizerFactory:
    @staticmethod
    def build(config: Union[AdamConfig, SGDConfig], model: torch.nn.Module) -> Optimizer:
        params = [model.parameters()]
        if isinstance(config, AdamConfig):
            return torch.optim.Adam(
                itertools.chain(*params), lr=config.lr, betas=config.betas, weight_decay=config.weight_decay
            )
        elif isinstance(config, SGDConfig):
            return torch.optim.SGD(
                itertools.chain(*params), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
            )
        else:
            raise NotImplementedError


class SchedulerFactory:
    @staticmethod
    def build(config: ImplementedSchedulers, optimizer: Optimizer):
        if isinstance(config, StepLRConfig):
            return lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
        else:
            raise NotImplementedError
