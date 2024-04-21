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
from hybrid_observer.training.loss.losses import Loss, StandardLoss, DataSimLoss, RegularizedDataSimLoss
from hybrid_observer.configs.training.training_config import (
    LossConfig,
    StandardLossConfig,
    DataSimLossConfig,
    SimAndDataRegularizedConfig,
)
from hybrid_observer.configs.training.enum import LossEnum
from hybrid_observer.basic_interfaces import Factory


def find_loss(loss: LossEnum) -> torch.nn.Module:
    if loss == LossEnum.RMSE:
        return torch.nn.functional.mse_loss
    if loss == LossEnum.RMAE:
        return torch.nn.functional.l1_loss


class LossFactory(Factory):
    @staticmethod
    def build(config: LossConfig) -> Loss:
        basic_loss = find_loss(config.loss_type)
        if config.loss_structure.__class__ == StandardLossConfig:
            return StandardLoss(basic_loss=basic_loss)
        elif config.loss_structure.__class__ == DataSimLossConfig:
            return DataSimLoss(basic_loss=basic_loss, sim_weight=config.loss_structure.sim_weight)
        elif config.loss_structure.__class__ == SimAndDataRegularizedConfig:
            return RegularizedDataSimLoss(
                basic_loss=basic_loss,
                sim_weight=config.loss_structure.sim_weight,
                reg_weight=config.loss_structure.reg_weight,
            )
        else:
            raise NotImplementedError
