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
from scipy.integrate import ode
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



def VDP(
    a: float = 5,
    b: float = 80,
    omega: float = 7,
    t: np.array = np.arange(0, 100, 0.05),
    initial_condition=[-2.0, 1.0, 0.31, 0.0],
):
    """
    build dataset
    ----------
    noise : float, optional
        noise level. The default is 0.00.
    a : float, optional
        left boarder. The default is 5.
    b : float, optional
        right boarder. The default is 80.

    Returns
    -------
    TYPE
        Union(np.array, np.array).
        noisy training data, noise-free ground truth, simulator data
    """

    def f(t, x_in):
        x = x_in[0]
        y = x_in[1]
        u = x_in[2]
        v = x_in[3]
        return [y, -x + a * (1 - x**2) * y + b * u, v, (omega**2) * (-u)]

    sol = solve_ivp(
        f,
        [0, 100],
        initial_condition,
        method="RK45",
        t_eval=t,
        rtol=10 ** (-5),
        atol=10 ** (-8),
        dense_output=False,
        events=None,
        vectorized=False,
        args=None,
    )
    sol = sol.y
    sol = np.transpose(sol)
    sol = sol.astype(np.float64)
    return sol[5:, 0]


def double_mass(
    t: np.array,
    omega_1: float = 0.5,
    omega_2: float = 4.0,
    a: float = 1.0,
    b: float = 0.0,
    phase_1: float = 0,
    phase_2: float = 0,
):
    """
    generates trajectories from a double mass-spring system specified by time t, frequencies omega_1, omega_2,
    amplitudes a and b and phase delays phase_1 and phase_2
    Args:
        t: time
        omega_1: first frequency
        omega_2: second frequency
        a: first apmlitude
        b: second amplitude
        phase_1: first phase delay
        phase_2: second phase delay

    Returns:
    trajectory of double mass-spring system at times t
    """
    sol_ref = a * np.cos(-omega_1 * t + phase_1) + b * np.cos(-omega_2 * t + phase_2)
    return sol_ref


def upswinging_behavior(t, omega_1, omega_2, a, b, phase_1, phase_2, damping):
    """
    build dataset
    ----------
    noise : float, optional
        noise level. The default is 0.00.
    a : float, optional
        left boarder. The default is 5.
    b : float, optional
        right boarder. The default is 80.
    lb : float, optional
         left integration border for scipy. The default is 0.
    ub : float, optional
        right integration border for scipy. The default is 100.
    dt : float, optional
        step size. The default is 0.05.

    Returns
    -------
    TYPE
        Union(np.array, np.array).
        noisy training data, noise-free ground truth, simulator data
    """

    solRef = a * np.cos(-omega_1 * t + phase_1) + b * np.exp(-damping * t) * np.cos(-omega_2 * t + phase_2)
    return np.float32(solRef)


