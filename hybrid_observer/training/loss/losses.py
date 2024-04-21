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
from torch import Tensor

from typing import Optional, Union, Callable


class Loss(torch.nn.Module):
    """
    computing loss between training data and batch, simulator and batch and a learned part based on
    the so called basic_loss which is MSE or MAE
    Args:
        basic_loss (torch.nn.Module): basic loss computation, here either MSE or MAE

    Attributes:
        basic_loss (torch.nn.Module): basic loss computation, here either MSE or MAE
    """

    def __init__(
        self,
        basic_loss: Union[torch.nn.functional.mse_loss, torch.nn.functional.l1_loss] = torch.nn.functional.mse_loss,
    ):
        super(Loss, self).__init__()
        self.basic_loss = basic_loss

    def forward(
        self,
        output: torch.Tensor,
        ref: torch.Tensor,
        sim_output: Optional[torch.Tensor],
        sim_ref: Optional[torch.Tensor],
        learned_part: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """

        Args:
            output (torch.Tensor): output corresponding to training data
            ref (torch.Tensor): reference values
            sim_output (Optional[torch.Tensor]): simulator outputs or None
            sim_ref (Optional[torch.Tensor]): simulator reference or None
            learned_part (Optional[torch.Tensor]): purely learning-based part

        Returns:
            torch.Tensor: loss(output, ref)

        """
        raise NotImplementedError


class StandardLoss(Loss):
    """
    computing loss between training data and batch, input is the so called basic_loss which is MSE or MAE
    Args:
        basic_loss (torch.nn.Module): basic loss computation, here either MSE or MAE

    Attributes:
        basic_loss (torch.nn.Module): basic loss computation, here either MSE or MAE
    """

    def __init__(
        self,
        basic_loss: Union[torch.nn.functional.mse_loss, torch.nn.functional.l1_loss] = torch.nn.functional.mse_loss,
    ):
        super(StandardLoss, self).__init__()
        self.basic_loss = basic_loss

    def forward(
        self,
        output: torch.Tensor,
        ref: torch.Tensor,
        sim_output: Optional[torch.Tensor],
        sim_ref: Optional[torch.Tensor],
        learned_part: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """

        Args:
            output (torch.Tensor): output corresponding to training data
            ref (torch.Tensor): reference values
            sim_output (Optional[torch.Tensor]): simulator outputs or None
            sim_ref (Optional[torch.Tensor]): simulator reference or None
            learned_part (Optional[torch.Tensor]): purely learning-based part

        Returns:
            torch.Tensor: loss(output, ref)

        """
        return self.basic_loss(output, ref)


class DataSimLoss(Loss):
    """
    computing loss between training data and batch, input is the so-called basic_loss which is MSE or MAE

    Attributes:
        basic_loss (Callable[[Tensor, Tensor], Tensor]): basic loss computation taking expected and target tensor as
            input and returning a loss value. Here either MSE or MAE
        sim_weight (float): regularizing weight for influence of simulator loss"""

    def __init__(
        self, basic_loss: Callable[[Tensor, Tensor], Tensor] = torch.nn.functional.mse_loss, sim_weight: float = 1
    ) -> torch.Tensor:
        """
        Args:
            basic_loss: basic loss computation taking expected and target tensor as
                input and returning a loss value. Here either MSE or MAE
            sim_weight: regularizing weight for influence of simulator loss
        """
        super(DataSimLoss, self).__init__()
        self.basic_loss = basic_loss
        self.sim_weight = sim_weight

    def forward(
        self,
        output: torch.Tensor,
        ref: torch.Tensor,
        sim_output: Optional[torch.Tensor],
        sim_ref: Optional[torch.Tensor],
        learned_part: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """

        Args:
            output (torch.Tensor): output corresponding to training data
            ref (torch.Tensor): reference values
            sim_output (Optional[torch.Tensor]): simulator outputs or None
            sim_ref (Optional[torch.Tensor]): simulator reference or None
            learned_part (Optional[torch.Tensor]): purely learning-based part

        Returns:
            torch.Tensor: loss(output, ref)+sim_weight*loss(sim, sim_ref)
        """
        return self.basic_loss(output, ref) + self.sim_weight * self.basic_loss(sim_output, sim_ref)


class RegularizedDataSimLoss(Loss):
    """
    computing loss between training data and batch, input is the so called basic_loss which is MSE or MAE
    Args:
        basic_loss (torch.nn.Module): basic loss computation, here either MSE or MAE
        sim_weight (float): regularizing weight for influence of simulator loss
        reg_weight (float): regularizing weight of the purely learning-based influence

    Attributes:
        basic_loss (torch.nn.Module): basic loss computation, here either MSE or MAE
        sim_weight (float): regularizing weight for influence of simulator loss
        reg_weight (float): regularizing weight of the purely learning-based influence
    """

    def __init__(
        self, basic_loss: torch.nn.Module = torch.nn.functional.mse_loss, sim_weight: float = 1, reg_weight: float = 1
    ):
        super(RegularizedDataSimLoss, self).__init__()
        self.basic_loss = basic_loss
        self.sim_weight = sim_weight
        self.reg_weight = reg_weight

    def forward(
        self,
        output: torch.Tensor,
        ref: torch.Tensor,
        sim_output: Optional[torch.Tensor],
        sim_ref: Optional[torch.Tensor],
        learned_part: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """

        Args:
            output (torch.Tensor): output corresponding to training data
            ref (torch.Tensor): reference values
            sim_output (Optional[torch.Tensor]): simulator outputs or None
            sim_ref (Optional[torch.Tensor]): simulator reference or None
            learned_part (Optional[torch.Tensor]): purely learning-based part

        Returns:
            torch.Tensor: loss(output, ref)+sim_weight*loss(sim, sim_ref)+reg_weight*loss(learned_part,0)
        """
        return (
            self.basic_loss(output, ref)
            + self.sim_weight * self.basic_loss(sim_output, sim_ref)
            + self.reg_weight * self.basic_loss(learned_part, torch.zeros(learned_part.shape))
        )
