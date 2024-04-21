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
import math

import numpy as np
import argparse
import pathlib

import torch

from hybrid_observer.datasets.generate_data import VDP 
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from os import getenv
import os
import matplotlib.pyplot as plt



def parse_args():
    parser = argparse.ArgumentParser(description="Process some configuration.")
    parser.add_argument("--a", help="first Van der Pol parameter", default=5.0)
    parser.add_argument("--b", help="second Van der Pol parameter", default=80)
    parser.add_argument("--omega", help="frequency of superposed oscillations", default=7.0)
    parser.add_argument(
        "--initial_val", help="provide the initial condition for the data", default=[-2.0, 1.0, 0.31, 0.0]
    )
    parser.add_argument("--lb", help="phase of first component", default=0.0)
    parser.add_argument("--ub", help="phase of first component", default=100.0)
    parser.add_argument("--noise", help="noise that is added to simulation", default=0.0)
    parser.add_argument("--index_delta", help="step size", default=0.1)
    return parser


def main():
    load_dotenv(find_dotenv())
    folder = Path(getenv("DATA_DIR"))
    if not folder.exists():
        folder.mkdir(parents=True)
    parser = parse_args()
    args = parser.parse_args()
    t = np.arange(args.lb, args.ub, args.index_delta)
    sol_ref = VDP(a=args.a, b=args.b, omega=args.omega, t=t, initial_condition=args.initial_val)
    sol = sol_ref + np.random.normal(size=[sol_ref.shape[0]]) * np.sqrt(args.noise)
    np.savez(Path.joinpath(folder, "Exp4.npz"), sol=sol, sol_ref=sol_ref, sim=sol, t=t[5:])
 
if __name__ == "__main__":
    main()
