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
#The datasets here are generated from data provided in here "Semiempirical identiﬁcation of nonlinear dynamics of a two-degree-of-freedom 
#real torsion pendulum with a nonuniformplanar stick–slip friction and elastic barriers" (Fig 6 and Fig 7)
#URL: https://link.springer.com/article/10.1007/s11071-020-05684-6
# Copyright (c) 2020 Bartłomiej Lisowski, Clement Retiere,
#José Pablo Garcia Moreno & Paweł Olejnik
#For license details, we refer to the 3rd-party-licenses.txt

import math

import numpy as np
import argparse
import pathlib

import torch

from hybrid_observer.utils import filter_via_torch, filter_values, pytorch_filters
from scipy.optimize import curve_fit

from hybrid_observer.datasets.generate_data import VDP, upswinging_behavior
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from os import getenv
import os
import matplotlib.pyplot as plt
from hybrid_observer.datasets.generate_data import upswinging_behavior
 



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
        help="provide path, where raw measurements are stored")
    parser.add_argument("--use_derivative", help="indicate whether position or derivative should be used", default=True)
    parser.add_argument("--a_distortion", help="indicate, whether distortion should be added", default=1.0)
    parser.add_argument("--frequency_distortion", help="frequency of distortion", default=50.0)
    parser.add_argument("--phase", help="provide phase shift of distortion", default=0.0)
    parser.add_argument("--damping", help="provide amount of damping", default=0.5)
    return parser


def main():
    load_dotenv(find_dotenv())
    folder = Path(getenv("DATA_DIR"))
    if not folder.exists():
        folder.mkdir(parents=True)
    parser = parse_args()
    args = parser.parse_args()

    sol_ref_velocity = np.loadtxt(args.data_location + "/" + "double_torsion_phi2.txt")
    sim_velocity = np.loadtxt(args.data_location + "/" + "double_torsion_num_phi2.txt")
    sol_ref_velocity, sim_velocity, t = normalize(sol_ref_velocity, sim_velocity)
    eps = upswinging_behavior(
        0.01 * t, 0.0, args.frequency_distortion, 0.0, args.a_distortion, 0.0, args.phase, args.damping
    )
    np.savez(
      Path.joinpath(folder, "Exp2.npz"),
      sol=sol_ref_velocity + eps,
      sol_ref=sol_ref_velocity + eps,
      sim=sim_velocity,
      t=t,
        )
    numerator, denominator = filter_values(0.1, "lowpass")
    lowpass = filter_via_torch(numerator, denominator, torch.Tensor(sim_velocity))
    np.savez(
        Path.joinpath(folder, "Exp2_Filter.npz"),
        sol=sol_ref_velocity + eps,
        sol_ref=sol_ref_velocity + eps,
        sim=lowpass.numpy(),
        t=t,
        )

if __name__ == "__main__":
    main()
