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
# Special layers 

import torch

from hybrid_observer.model.interfaces import InvertibleLayer
from typing import Optional

# Implementations of invertible transformations to directly obtain the dynamics.


class AffineCoupling(InvertibleLayer):
    """Affine Coupling layer as proposed in Dinh et. al. 2014 (NICE: Non-linear Indendepent Components Estimation.)
    Args:
        s (torch.nn.Module): neural network
        t (torch.nn.Module): neural network
    Attributes:
        s (torch.nn.Module): neural network
        t (torch.nn.Module): neural network
    """

    def __init__(self, s: torch.nn.Module(), t:torch.nn.Module()):
        super(AffineCoupling, self).__init__()
        self.s = s
        self.t = t

    def forward(self, x: torch.Tensor):
        """
        forward propagation through invertible layers
        """
        xa, xb = torch.chunk(x, 2, dim=2)
        log_s = self.s(xb)
        s = torch.exp(log_s)
        ya = s * xa + self.t(xb)
        yb = xb
        return torch.cat([ya, yb], dim=2)

    def inverse(self, y: torch.Tensor):
        """
        inverse propagation through inverse layer
        """
        ya, yb = torch.chunk(y, 2, dim=2)
        log_s = self.s(yb)
        s = torch.exp(log_s)
        xa = (ya - self.t(yb)) / s
        xb = yb
        return torch.cat([xa, xb], dim=2)


class LayerList(InvertibleLayer):
    """Stacking Multiple Invertible Layers.
    Args:
        layers (List(torch.nn.Modules): List of invertible layers
    Attributes:
        layers (List(torch.nn.Modules): List of invertible layers
    """

    def __init__(self, list_of_layers: Optional[list[InvertibleLayer]] = None):
        super(LayerList, self).__init__()
        self.layers = torch.nn.ModuleList(list_of_layers)

    def __getitem__(self, i: int):
        return self.layers[i]

    def forward(self, x: torch.Tensor):
        """
        forward propagation
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def inverse(self, x: torch.Tensor):
        """
        inverse propagation
        """
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x
