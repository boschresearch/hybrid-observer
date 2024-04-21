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
# The data used in this dataset can be generated with the code provided in
# https://github.com/Open-Source-Drilling-Community/Aarsnes-and-Shor-Torsional-Model 
# with the function (main.m) (Copyright (c) 2021 Open Drilling Project).
# The data are also represented in the paper: Aarsnes, Ulf Jakob F., and Roman J. Shor. 
# Torsional vibrations with bit off bottom: Modeling, characterization and field data validation." 
# Journal of Petroleum Science and Engineering 163 (2018): 712-721.
# For license details, we refer to 3rd-party-licenses.txt"

import math

import numpy as np
import argparse
import pathlib

import torch
from scipy.optimize import curve_fit

from hybrid_observer.datasets.generate_data import VDP
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from os import getenv
import os
import matplotlib.pyplot as plt

from hybrid_observer.datasets.generate_data import upswinging_behavior
from scipy.fft import fft, ifft, fftfreq
from hybrid_observer.utils import filter_via_torch, filter_values, pytorch_filters



def normalize(sol_ref: np.ndarray, sim:np.ndarray):
    """
    normalize data
    """
    num_data = sim.shape[0]
    t = np.arange(0, num_data, 1)
    sol_ref = sol_ref[:num_data]
    mean = sum(sol_ref) / num_data
    std = np.sqrt(sum((sol_ref - mean) ** 2) / (num_data - 1))
    sol_ref = (sol_ref - mean) / std
    sim = (sim - mean) / std
    return sol_ref, sim, t


def parse_args():
    parser = argparse.ArgumentParser(description="Process some configuration.")
    parser.add_argument(
        "--data_location",
        help="provide path, where raw measurements are stored"
    )
    return parser


def main():
    load_dotenv(find_dotenv())
    folder = Path(getenv("DATA_DIR"))
    if not folder.exists():
        folder.mkdir(parents=True)
    parser = parse_args()
    args = parser.parse_args()
    sol_ref = np.loadtxt(args.data_location + "/" + "drill_string_data.txt")
    sim = np.loadtxt(args.data_location + "/" + "sim.txt")
    sol_ref, sim, t = normalize(sol_ref, sim)
    numerator, denominator = filter_values(0.05, "lowpass")
    lowpass = filter_via_torch(numerator, denominator, torch.Tensor(sim))
    np.savez(
        Path.joinpath(folder, "Exp3.npz"),
        sol=sol_ref[2000:],
        sol_ref=sol_ref[2000:],
        sim=sim[2000:],
        t=t)
    np.savez(
        Path.joinpath(folder, "Exp3_Filter.npz"),
        sol=sol_ref[2000:],
        sol_ref=sol_ref[2000:],
        sim=lowpass[2000:].numpy(),
        t=t,
    )


if __name__ == "__main__":
    main()
