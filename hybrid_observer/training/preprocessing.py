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

from typing import Optional
import torch
import torchaudio
import numpy as np
from scipy.signal import iirfilter


class IIRValues:
    """IIR-values for IIR-filter.

    Args:
        N (int): filter order.
        Wn (torch.double): cutoff frequency
        btype (str): type of filter ("lowpass", "highpass")
        ftype (str): type of filter ("butter", ...)
        fs (str): 1/dt


    Attributes:
        N (int): filter order.
        Wn (torch.double): cutoff frequency
        btype (str): type of filter ("lowpass", "highpass")
        ftype (str): type of filter ("butter", ...)
        fs (str): 1/dt

    """

    def __init__(
        self, N: int = 1, Wn: torch.double = 0.1, btype: str = "lowpass", ftype: str = "butter", fs: torch.double = 10
    ):
        super().__init__()
        self.N = N
        self.Wn = Wn
        self.btype = btype
        self.ftype = ftype
        self.fs = fs

    def filterValues(self):
        """output of scipy filter: b, a with G(z)=\sum_k b_k z^{-k}/ a_k z^{-k}

        Returns
        -------
        denominator : float
            b.
        numerator : float
            a.

        """
        denominator, numerator = iirfilter(
            N=self.N,
            Wn=self.Wn,
            rp=None,
            rs=None,
            btype=self.btype,
            analog=False,
            ftype=self.ftype,
            output="ba",
            fs=self.fs,
        )
        numerator = torch.from_numpy(numerator)
        denominator = torch.from_numpy(denominator)
        return denominator, numerator


def filter_via_torch(numerator: torch.Tensor, denominator: torch.Tensor, X: torch.Tensor):
    """
    filters signal with forward-backward method (without delay) with torch filters
    filter corrresponds to IIR filter with transition function H = \sum_k a_k/\sum_k b_k
    ----------
    numerator : (n,1)-Tensor
        numerator of polynomial coefficients
    denominator : (n,1)-Tensor
        denominator of polynomial coefficients.
    X : (batchsize, timesteps)-Tensor.

    Returns
    -------
    result : (batchsize, timesteps)-Tensor
        filtered signal.

    """
    max_X = 1
    if torch.max(torch.sqrt(X**2)) > 1:
        max_X = torch.max(torch.sqrt(X**2))
    filtered_X = torchaudio.functional.filtfilt(X / max_X, numerator.float(), denominator.float(), False)
    result = filtered_X * max_X
    return result


def pytorch_filters(
    solRef: np.array, numerator: np.array, denominator: np.array, numerator_tilde: np.array, denominator_tilde: np.array
):
    """preprocess ground truth with torch filters"""
    lowpass = filter_via_torch(numerator, denominator, torch.FloatTensor(solRef[:, 0]))
    highpass = filter_via_torch(numerator_tilde, denominator_tilde, torch.FloatTensor(solRef[:, 0]))
    return lowpass, highpass


def filter_values(order: int = 1, w: float = 0.1, mode: str = "lowpass", type: str = "butter"):
    """
    compute IIR filter values with specified parameters via scipy filter design
    H (z)= \sum_k a_k z^{n-1}/ \sum_k b_k z^{n-k}
    ----------
    w : float
        cutoff frequency.
    mode : string out of "lowpass", "highpass", "bandpass"
        type of filter.
    device : torch.device
        device.

    Returns
    -------
    numerator : (n,1)-numpy array
         a_k coefficients.
    denominator : (n,1)- numpy array
        b_k coefficients.

    """
    IIR = IIRValues(order, w, mode, type, 10)
    denominator, numerator = IIR.filterValues()
    return numerator, denominator


class Preprocessor:
    """
    interface for Preprocessor
    """

    def preprocess(self, training_data):
        raise NotImplementedError


class Filter(Preprocessor):
    """
    preprocess data with IIR filter and downsampling
    """

    def __init__(
        self,
        order: int = 1,
        cutoff: float = 0.07,
        mode: str = "lowpass",
        type: str = "butter",
        downsampling_rate: int = 2,
    ):
        super().__init__()
        self.order = order
        self.cutoff = cutoff
        self.mode = mode
        self.type = type
        self.downsampling_rate = downsampling_rate

    def preprocess(self, training_data: torch.Tensor):
        """
        Preprocess training data
        """
        numerator, denominator = filter_values(self.order, self.cutoff, self.mode, self.type)
        lowpass = filter_via_torch(numerator, denominator, training_data)
        return lowpass[:: self.downsampling_rate]


class Postprocessor:
    """
    Postprocessor (usually inverse operation of Preprocessor)
    """

    def postprocess(self, data):
        """
        postprocess
        """
        raise NotImplementedError


class Upsampler(Postprocessor):
    """
    Postprocessing via Upsampling
    """

    def __init__(
        self,
        upsampling_rate: int = 2,
    ):
        super().__init__()
        self.m = torch.nn.Upsample(scale_factor=upsampling_rate, mode="linear", align_corners=False)

    def postprocess(self, output):
        """
        Upsample smoothly
        """
        return self.m(output.permute(2, 0, 1)).permute(0, 2, 1)
