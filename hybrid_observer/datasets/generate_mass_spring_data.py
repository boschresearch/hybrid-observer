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

import numpy as np
import argparse
import pathlib
from hybrid_observer.datasets.generate_data import double_mass
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from os import getenv
import os
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Process some configuration.")
    parser.add_argument("--omega_1", help="frequency of first component", default=0.8)
    parser.add_argument("--omega_2", help="frequency of second component", default=4.0)
    parser.add_argument("--a", help="amplitude of first component", default=1.0)
    parser.add_argument("--b", help="amplitude of second component", default=0.5)
    parser.add_argument("--phase_1", help="phase of first component", default=0.0)
    parser.add_argument("--phase_2", help="phase of first component", default=0.0)
    parser.add_argument("--omega_1_sim", help="frequency of first component", default=0.8)
    parser.add_argument("--omega_2_sim", help="frequency of second component", default=4.0)
    parser.add_argument("--a_sim", help="amplitude of first component", default=0.2)
    parser.add_argument("--b_sim", help="amplitude of second component", default=0.0)
    parser.add_argument("--phase_1_sim", help="phase of first component", default=0.0)
    parser.add_argument("--phase_2_sim", help="phase of first component", default=0.0)
    parser.add_argument("--noise", help="phase of first component", default=0.1)
    parser.add_argument("--lb", help="phase of first component", default=0.0)
    parser.add_argument("--ub", help="phase of first component", default=100.0)
    parser.add_argument("--index_delta", help="phase of first component", default=0.1)
    return parser


def main():
    load_dotenv(find_dotenv())
    folder = Path(getenv("DATA_DIR"))
    if not folder.exists():
        folder.mkdir(parents=True)
    parser = parse_args()
    args = parser.parse_args()
    t = np.arange(args.lb, args.ub, args.index_delta)
    sol_ref = double_mass(t, args.omega_1, args.omega_2, args.a, args.b, args.phase_1, args.phase_2)
    sim = double_mass(t, args.omega_1_sim, args.omega_2_sim, args.a_sim, args.b_sim, args.phase_1_sim, args.phase_2_sim)
    sol = sol_ref + np.random.normal(size=[sol_ref.shape[0]]) * np.sqrt(args.noise)
    np.savez(Path.joinpath(folder, "Exp1.npz"), sol=sol, sol_ref=sol_ref, sim=sim, t=t)


if __name__ == "__main__":
    main()
