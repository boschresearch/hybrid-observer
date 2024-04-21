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
from typing import Optional, List, Tuple

from torch import Tensor
from torch.nn import Module


class InvertibleLayer(torch.nn.Module):
    """
    Invertible layer contains forward and inverse method
    """

    def forward(self, x: torch.Tensor):
        raise NotImplementedError

    def inverse(self, y: torch.Tensor):
        raise NotImplementedError


class TrainableSimulator(torch.nn.Module):
    """
    Trainable Simulator is usually a parametric model with trainable parameters (stationary signal, dynamical system)
    or simply returns unmodified simulator signal if simulator is not trainable.
    """

    def forward(self, t: torch.Tensor, s: torch.Tensor):
        raise NotImplementedError


class Observer(torch.nn.Module):
    """interface for observer model"""

    def get_simulator_trajectory(self, u: torch.Tensor) -> torch.Tensor:
        """
        returns the output of the simulator observation model h to z.
        Interface can also be adapted to the naive hybrid scenario by returning 1-dimensional latent state as output
        Args:
            u ((batchsize, horizon, d_u)-torch.Tensor): latent trajectory of observable states.
              Note: d_u = 1 for naive case

        Returns:
            ((batchsize, horizon, d_s)-torch.Tensor): reconstructed sim outputs.

        """
        raise NotImplementedError

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s ((batchsize, horizon, d_s)-torch.Tensor): simulator trajectory.

        Returns:
            ((batchsize, horizon, d_y)-torch.Tensor): mapping of
            observable states (corresponds to g_{\theta}(u_n) in Eq. 12.
            ((batchsize, horizon, d_u)-torch.Tensor): observable states.
            Note: in naive setting u is one-dimensional and equals outputs
        """
        raise NotImplementedError


class RNN(torch.nn.Module):
    """interface for standard RNN prediction with warmup phase, output feedback
    and control input
    (standard RNN, LSTM, GRU...)
    """

    def initialize(self):
        """
        reset hidden states
        """
        raise NotImplementedError

    def forward(self, y: torch.Tensor, u: Optional[torch.Tensor]) -> torch.Tensor:
        """
        one-step ahead predictions of RNN dynamics
        Args:
            u ((batchsize, 1, d_u)-torch.Tensor): control inputs (here observable
              states u).
            y ((batchsize, 1, d_y)-torch.Tensor): observations y_hat.

        Returns:
            ((batchsize, 1, output_dimension)-torch.Tensor.
            Note: returns predictions
        """
        raise NotImplementedError


class TSModel(torch.nn.Module):
    """
    this is a model to predict time series given observational and/or simulator data.

    Note: TS Model fits the training method in general.
          However, here we use TimeDependentTSModel for configuration.
          It is currently the only registered model.
    """

    def fit(self, batch: torch.Tensor, loss: torch.nn.Module) -> torch.Tensor:
        """
        computes a step of the loss between batch
        batch needs to have the following form: batch[0]: data, batch[1]: simulator trajectory

        """
        raise NotImplementedError

    def rollout(
        self, s: torch.Tensor, y_n: torch.Tensor, t: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        computes output of residual model (Eq. 11)
        Args:
            s ((batchsize, rollout_length, dim)-torch.Tensor): simulator trajectory.
            y_n ((batchsize, recognition_length, dim)-torch.Tensor): short part of training trajectory for recognition.
            t ((batchsize, rollout_length, 1): torch.Tensor): time vector.
        Returns:
            simulator_trajectory ((batchsize, rollout_length, simulator_dim)-torch.Tensor)): simulator reconstructions if available.
            learned_trajectory ((batchsize, rollout_length, dim)-torch.Tensor): learning-based residuum if available.
            full_trajectory ((batchsize, rollout_length, dim)-torch.Tensor)): output (to reproduce training data) full output.

        """
        raise NotImplementedError

    def rollout_wrapper(
        self, test_X: List[Tensor]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        wraps the evaluation data such that they fit the rollout
        """
        raise NotImplementedError


class Rollouts(torch.nn.Module):
    """
    Producing trajectory with RNN dynamics
    """

    def get_recognition_batch(self, y: torch.Tensor) -> torch.Tensor:
        """
        cut recongition batch from trajectory to get proper warmup

        Args:
            y ((batchsize, trajectory_length, dim)-torch.Tensor):training trajectory

        Returns:
            y_recognition ((batchsize, recognition_steps, dim)-torch.Tensor):training trajectory

        """
        raise NotImplementedError

    def warmup(self, y: torch.Tensor, u: Optional[torch.Tensor]):
        """
        warmup by feedbacking observations instad of outputs
        Args:
            u ((batchsize, warmup_length, d_u)-torch.Tensor): control inputs (here observable
              states u).
            y ((batchsize, warmup_length, d_y)-torch.Tensor): observations y_hat.

        Returns:
            None.
            Note: in-place adaption of hidden state

        """
        raise NotImplementedError

    def rollout(
        self, y_recognition: torch.Tensor, control_input: Optional[torch.Tensor], t: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, None, None]:
        """

        Args:
            y_recognition ((batchsize, recongition_length, d_x)-torch.Tensor): first observations for recognition model.
            control_input ((batchsize, trajectory_length, d_u)-torch.Tensor): control input (here: observable states).
            t ((batchsize, trajectory_length,1)-torch.Tensor: time vector
        Returns:
            X ((batchsize, recongition_length, d_y)-torch.Tensor): output trajectory.

        """
        raise NotImplementedError
