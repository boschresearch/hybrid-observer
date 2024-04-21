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

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from torch import Tensor

from hybrid_observer.model.layers import AffineCoupling, LayerList
from hybrid_observer.utils import filter_values, filter_via_torch
from hybrid_observer.model.interfaces import Observer, RNN, Rollouts, TSModel, TrainableSimulator, InvertibleLayer


class UnmodifiedSimulator(TrainableSimulator):
    """
    Wrap the untrainable simulator in a trainable simulator by simply returning the simulator output again.
    """

    def forward(self, t:torch.Tensor, s:torch.Tensor):
        return s


class TrainableSineSimulator(TrainableSimulator):
    """
    Trainable sine simulator returns a stationary sine signal with trainable parameters.
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        frequency: float = 0.007,
        frequency_shift: float = 0.0,
    ):
        super(TrainableSineSimulator, self).__init__()
        self.amplitude = nn.Parameter(torch.Tensor([amplitude]), requires_grad=True)
        self.frequency = nn.Parameter(torch.Tensor([frequency]), requires_grad=True)
        self.frequency_shift = nn.Parameter(torch.Tensor([frequency_shift]), requires_grad=True)

    def forward(self, t:torch.Tensor, s:torch.Tensor):
        """
        Args:
            t: ((batchsize, trajectory_length, 1)-torch.Tensor: time.
            s: placeholder to fit interface.

        """
        return self.amplitude * torch.sin(2 * math.pi * t * self.frequency + self.frequency_shift)


class FilteredSimulator(TrainableSimulator):
    """
    Trainable simulator with additional filtering 
    """
    def __init__(self, Wn=0.1, btype="highpass", trainable_simulator: TrainableSimulator = TrainableSineSimulator):
        super(FilteredSimulator, self).__init__()
        self.numerator, self.denominator = filter_values(Wn, btype)
        self.trainable_simulator = trainable_simulator

    def forward(self, t:torch.Tensor, s:torch.Tensor) -> torch.Tensor:
        filtered_trajectory = filter_via_torch(
            self.numerator, self.denominator, self.trainable_simulator(t, s).squeeze(2)
        ).unsqueeze(2)
        return filtered_trajectory


class BasicSimulator(Observer):
    """
    simulator trajectory wrapped in the observer interface, receives sim trajectory as an input, provides sim trajectory
    as an output.
    """

    def __init__(self):
        super(BasicSimulator, self).__init__()

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            s ((batchisze, trajectory_length, dim)-torch.Tensor: simulator trajectory.

        Returns:
            s (see input).
            s (see input).
        """
        return s, s, s

    def get_simulator_trajectory(self, s: torch.Tensor) -> torch.Tensor:
        """

        Args:
            s ((batchisze, trajectory_length, dim)-torch.Tensor: simulator trajectory.

        Returns:
            s (see input).
        """
        simulator_outputs = s
        return simulator_outputs


class KKL(Observer):
    """KKL Observer model receives simulator trajectory as an input and provides latent state and observation reconstruction as an output.
       The choice of the MLP as transformation network is proposed in Janny et. al. ("Deep KKL: Data-driven Output Prediction for Non-Linear Systems")

    Args:
        simulator_dim (int): dimension of simulator observations.
        observation_dim (int): dimension of observations.
        latent_space_dim (int): dimension of the observable latent states.
        mlp_hidden_dim (int): dimension of the hidden MLP layers.
        device (torch.device): device.
        invertible_dynamics (bool): use invertible NN?

    Attributes:
        latent_dim (int): dimension of the observable latent states.
        D (torch.nn.Parameter): prediction matrix(latent_dim, latent_dim) (cf. Eq. 9).
        F (torch.nn.Parameter): prediction matrix(latent_dim, dim) (cf. Eq. 9).
        T_star (torch.nn.Sequential): transformation MLP (cf. Eq. 9).
        h (torch.nn.Linear): simulator Observer model (cf. Eq. 10).
        g_s (torch.nn.Linear): linear mapping of observable states (cf. Eq. 10).
        device (torch.device): device.
        invertible_dynamics (bool): use invertible NN?

    """

    def __init__(
        self,
        simulator_dim: int = 1,
        observation_dim: int = 1,
        latent_space_dim: int = 4,
        mlp_hidden_dim: int = 100,
        device: Optional[str] = "cpu",
        invertible_dynamics: bool = False,
    ):
        super(KKL, self).__init__()
        self.latent_dim = latent_space_dim
        self.sig = torch.nn.Sigmoid()
        self.D = nn.Parameter(torch.diag(self.sig(torch.randn(latent_space_dim))), requires_grad=True)
        self.register_buffer("F", torch.ones(latent_space_dim, simulator_dim))
        self.invertible_dynamics = invertible_dynamics
        if invertible_dynamics:
            self.T_star = LayerList(
                [
                    AffineCoupling(
                        nn.Sequential(nn.Linear(latent_space_dim // 2, latent_space_dim // 2), torch.nn.ReLU()),
                        nn.Sequential(nn.Linear(latent_space_dim // 2, latent_space_dim // 2), torch.nn.ReLU()),
                    ),
                    AffineCoupling(
                        nn.Sequential(nn.Linear(latent_space_dim // 2, latent_space_dim // 2), torch.nn.ReLU()),
                        nn.Sequential(nn.Linear(latent_space_dim // 2, latent_space_dim // 2), torch.nn.ReLU()),
                    ),
                ]
            )
        else:
            self.T_star = nn.Sequential(
                nn.Linear(latent_space_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, latent_space_dim),
            )
        self.h = nn.Linear(latent_space_dim, simulator_dim)
        self.g_s = nn.Linear(latent_space_dim, observation_dim)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def n_step_prediction(self, s: torch.Tensor):
        """
        n step prediction, where length corresponds to simulator length 
        Args:
            s ((batchsize, horizon, d_s)-torch.Tensor): simulator input.

        Returns:
            u ((batchsize, horizon, d_u)-torch.Tensor): latent states.

        """

        batchsize, horizon, d_s = s.shape
        z = torch.zeros(batchsize, self.latent_dim)
        latent_values = []
        # rollout
        for n in range(horizon):
            latent_values.append(z)
            z = self.dynamic(z, s[:, n])
        latent_values = torch.stack(latent_values, dim=1)
        u = self.T_star(latent_values)
        return u

    def invertible_n_step_prediction(self, s: torch.Tensor):
        """
        computes the invertible dynamics corresponding to f_u if transformation is given by an invertible network
        Args:
            s ((batchsize, horizon, d_s)-torch.Tensor): simulator input.

        Returns:
            u ((batchsize, horizon, d_u)-torch.Tensor): latent states.

        """
        batchsize, horizon, d_s = s.shape
        u = torch.zeros(batchsize, self.latent_dim)
        z = self.T_star.inverse(u.unsqueeze(1)).squeeze(1)
        latent_values = []
        for n in range(horizon):
            latent_values.append(u)
            z = self.dynamic(z, s[:, n])
            u = self.T_star(z.unsqueeze(1)).squeeze(1)
        u = torch.stack(latent_values, dim=1)
        return u

    def dynamic(self, z: torch.Tensor, s: Optional[torch.Tensor]):
        """
        returns one transition of the linear dynamical system z_{n+1}=Dz_n+Fs_n 
        Args:
            z ((batchsize, d_z)-torch.Tensor): KKL-states z 
            s ((batchsize, d_s)): simulator.

        Returns:
            z_new ((batchsize, d_z)-torch.Tensor): next step in trajectory.

        """
        z_new = torch.zeros_like(z)
        inds_finite = torch.isfinite(s[:, 0])
        inds_nan = torch.logical_not(inds_finite)
        z_new[inds_finite] = (
            torch.matmul(self.D, z[inds_finite].unsqueeze(-1)) + torch.matmul(self.F, s[inds_finite].unsqueeze(-1))
        ).squeeze(-1)
        if torch.any(inds_nan):
            z_new[inds_nan] = (
                torch.matmul(self.D, z[inds_nan].unsqueeze(0))
                + torch.matmul(self.F, (self.get_simulator_trajectory(self.T_star(z[inds_nan]))).unsqueeze(-1))
            ).squeeze(-1)
        return z_new

    def get_simulator_trajectory(self, u: torch.Tensor) -> torch.Tensor:
        """
        returns the output of the simulator observation model h to z.
        Args:
            u ((batchsize, horizon, d_u)-torch.Tensor): latent trajectory of observable states.

        Returns:
            simulator_outputs ((batchsize, horizon, d_s)-torch.Tensor): reconstructed sim outputs.

        """

        simulator_outputs = self.h(u)
        return simulator_outputs

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s ((batchsize, horizon, d_s)-torch.Tensor): simulator trajectory.

        Returns:
            projected_outputs ((batchsize, horizon, d_y)-torch.Tensor): mapping of
            observable states (corresponds to g_{\theta}(u_n))
            u ((batchsize, horizon, d_u)-torch.Tensor): observable states.

        """
        if self.invertible_dynamics:
            u = self.invertible_n_step_prediction(s)
        else:
            u = self.n_step_prediction(s)
        projected_outputs = self.g_s(u)
        return projected_outputs, u, s


class GRUModel(RNN):
    """GRU model for time series generation.

    produces GRU outputs via h_{n+1}=GRU(h_n, y_n), out_n = linear(h_n),
    meaning that the hidden layers receive the previous output of the linear
    observation model as an additional input. The warmup phase is performed by
    instead providing the observations $\hat{y}_n$ as an input.

    Args:
        input_dim (int): input dimension
        hidden_dim (int): number of nodes in each layer.
        output_dim(int): output dimension
        devive (torch.device): device

    Attributes:
        hidden_dim (int): number of nodes in each layer.
        h (None): current value of hidden layer.
        gru (torch.nn.GRU): torch gru layer
        fc (torch.nn.Linear): linear layer
        fl (torch.nn.Linear): linear layer
        device:torch.device: device

    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 1, device: Optional[str] = "cpu"):
        super(GRUModel, self).__init__()
        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.h = None
        # GRU layers
        self.gru = torch.nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    @staticmethod
    def prepare_inputs(y: torch.Tensor, sim: Optional[torch.Tensor]) -> torch.Tensor:
        if sim is not None:
            if y is not None:
                inputs = torch.cat([y, sim], 2)
            else:
                inputs = sim
        else:
            inputs = y
        return inputs

    def one_step(self, inputs: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(inputs, self.h)
        self.h = hidden
        return hidden

    def initialize(self):
        """
        initialize hidden states
        """
        self.h = None

    def forward(self, y: torch.Tensor, sim: Optional[torch.Tensor]) -> torch.Tensor:
        """
        one-step ahead predictions of GRU dynamics
        Args:
            u ((batchsize, 1, d_u)-torch.Tensor): control inputs (here observable
              states u).
            y ((batchsize, 1, d_y)-torch.Tensor): observations y_hat.
            sim ((batchsize, 1, d_sim)-torch.Tensor): simulator (or in general control input) trajectory
        Returns:
            out: ((batchsize, 1, output_dimension)-torch.Tensor.
        """

        inputs = self.prepare_inputs(y, sim)
        hidden = self.one_step(inputs)
        out = self.fc(hidden.permute(1, 0, 2))
        return out


class RNNRollouts(Rollouts):
    """Producing trajectory with consistent GP dynamics with n steps
    Args:
        rnn_one_step (torch.nn.Module): RNN-module that produces one step ahead predictions
        recognition_steps (int): number of steps for the warmup phase
        data_as_input (bool): use data in warmup phase? 
    Attributes:
        rnn_one_step (torch.nn.Module): RNN-module that produces one step ahead predictions
        recognition_steps (int): number of steps for the warmup phase
        data_as_input (bool): use data in warmup phase? 
    """

    def __init__(
        self, rnn_one_step: Optional[torch.nn.Module] = None, recognition_steps: int = 50, data_as_input: bool = True
    ):
        super().__init__()
        self.rnn_one_step = rnn_one_step if rnn_one_step is not None else GRUModel()
        self.recognition_steps = recognition_steps
        self.data_as_input = data_as_input

    def get_recognition_batch(self, y: torch.Tensor) -> torch.Tensor:
        """
        cut recongition batch from trajectory to get proper warmup
        Args:
            y ((batchsize, trajectory_length, dim)-torch.Tensor):training trajectory
        Returns:
            y_recongition ((batchsize, recognition_steps, dim)-torch.Tensor):training trajectory
        """
        y_recognition = y[:, 0 : self.recognition_steps, :]
        return y_recognition

    def warmup(self, y: torch.Tensor, u: Optional[torch.Tensor]):
        """
        Args:
            u ((batchsize, warmup_length, d_u)-torch.Tensor): control inputs (here observable
              states u).
            y ((batchsize, warmup_length, d_y)-torch.Tensor): observations y_hat.
        Returns:
            None.
        """

        steps = y.shape[1] if y is not None else u.shape[1]
        inputs = self.rnn_one_step.prepare_inputs(y, u)
        for j in range(steps):
            _ = self.rnn_one_step.one_step(inputs[:, j : j + 1, :])

    def rollout(
        self, y: torch.Tensor, control_input: Optional[torch.Tensor], t: torch.Tensor, nSteps: int
    ) -> Tuple[torch.Tensor, None, None]:
        """
        Args:
            y_recognition ((batchsize, trajectory_length, d_x)-torch.Tensor): training data.
            control_input ((batchsize, trajectory_length, d_u)-torch.Tensor): control input (here: observable states).
        Returns:
            X ((batchsize, recongition_length, d_y)-torch.Tensor): output trajectory.
            None: fit interface.
            None: fit interface.
        """

        y_recognition = self.get_recognition_batch(y)
        self.rnn_one_step.initialize()
        recognition_control = control_input[:, 0 : self.recognition_steps, :] if control_input is not None else None
        self.warmup(y_recognition, recognition_control) if self.data_as_input else self.warmup(
            None, recognition_control
        )
        x = y_recognition[:, -1, :].unsqueeze(1)
        X = [x]
        x_shape = x.shape
        for j in range(self.recognition_steps + 1, nSteps):
            control = control_input[:, j : j + 1, :] if control_input is not None else None
            x = (
                self.rnn_one_step(x, control).view(x_shape)
                if self.data_as_input
                else self.rnn_one_step(None, control).view(x_shape)
            )
            X.append(x)
        X = torch.cat(X, 1)
        X = torch.cat((y_recognition, X), 1)
        return X, None, None


class SummedRNNRollouts(RNNRollouts):
    """Rollouts that consist of the sum of two GRUs (observer_gru and residuum_gru)
    Args:
        rnn_one_step_observer (Optional[torch.nn.Module]): RNN serving as observer,
        rnn_one_step_residuum (Optional[torch.nn.Module]): RNN serving as residuum,
        recognition_steps (int): length of warmup phase,
        data_as_input_observer (bool): indicating if observer receives data,
        data_as_input_residuum (bool): indicating if residuum receives data,
        simulator_dim (int): dimension of simulator,
        simulator_as_input_residuum (bool): indicating if residuum receives simulator as input
    Attributes:
        rnn_one_step_observer (Optional[torch.nn.Module]): RNN serving as observer,
        rnn_one_step_residuum (Optional[torch.nn.Module]): RNN serving as residuum,
        recognition_steps (int): length of warmup phase,
        data_as_input_observer (bool): indicating if observer receives data,
        data_as_input_residuum (bool): indicating if residuum receives data,
        fc (torch.nn.Linear): linear model for simulator observations,
        simulator_dim (int): dimension of simulator,
        simulator_as_input_residuum (bool): indicating if residuum receives simulator as input     """

    def __init__(
        self,
        rnn_one_step_observer: Optional[torch.nn.Module] = GRUModel(),
        rnn_one_step_residuum: Optional[torch.nn.Module] = GRUModel(),
        recognition_steps: int = 50,
        data_as_input_observer: bool = True,
        data_as_input_residuum: bool = True,
        simulator_dim=1,
        simulator_as_input_residuum=False,
    ):
        super().__init__()
        self.rnn_one_step_observer = rnn_one_step_observer
        self.rnn_one_step_residuum = rnn_one_step_residuum
        self.recognition_steps = recognition_steps
        self.data_as_input_observer = data_as_input_observer if data_as_input_observer is not None else True
        self.data_as_input_residuum = data_as_input_residuum if data_as_input_residuum is not None else True
        self.fc = torch.nn.Linear(rnn_one_step_observer.hidden_dim, simulator_dim)
        self.simulator_as_input_residuum = simulator_as_input_residuum

    def warmup(self, y: torch.Tensor, u: Optional[torch.Tensor]):
        """
        Args:
            u ((batchsize, warmup_length, d_u)-torch.Tensor): control inputs (here observable
              states u).
            y ((batchsize, warmup_length, d_y)-torch.Tensor): observations y_hat.
        Returns:
            None.
        """
        rnn_one_steps = [self.rnn_one_step_observer, self.rnn_one_step_residuum]
        simulator_as_inputs = [True, self.simulator_as_input_residuum]
        data_as_inputs = [self.data_as_input_observer, self.data_as_input_residuum]
        for num_rnns in range(0, len(rnn_one_steps)):
            rnn_one_step = rnn_one_steps[num_rnns]
            if rnn_one_step is not None:
                simulator_as_input = simulator_as_inputs[num_rnns]
                data_as_input = data_as_inputs[num_rnns]
                steps = y.shape[1] if y is not None else u.shape[1]
                y_transformed = y if data_as_input else None
                inputs = (
                    rnn_one_step.prepare_inputs(y_transformed, u)
                    if simulator_as_input
                    else rnn_one_step.prepare_inputs(y_transformed, None)
                )
                for j in range(steps):
                    _ = rnn_one_step.one_step(inputs[:, j : j + 1, :])

    def get_simulator(self, x_observer: torch.Tensor):
        """
        produce simulator out of latent states
        """
        out = self.fc(x_observer.permute(1, 0, 2))
        return out

    def rollout(self, y: torch.Tensor, control_input: Optional[torch.Tensor], t: torch.Tensor, nSteps: int):
        """
        Args:
            y_recognition ((batchsize, trajectory_length, d_x)-torch.Tensor): training data.
            control_input ((batchsize, trajectory_length, d_u)-torch.Tensor): control input (here: observable states).
        Returns:
            X ((batchsize, recongition_length, d_y)-torch.Tensor): output trajectory.
            None: fit interface.
            None: fit interface.
        """

        y_recognition = self.get_recognition_batch(y)
        for rnn_one_step in [self.rnn_one_step_observer, self.rnn_one_step_residuum]:
            if rnn_one_step is not None:
                rnn_one_step.initialize()
        recognition_control = control_input[:, 0 : self.recognition_steps, :] if control_input is not None else None
        self.warmup(y_recognition, recognition_control) if self.data_as_input else self.warmup(
            None, recognition_control
        )
        x = y_recognition[:, -1, :].unsqueeze(1)
        X_simulator = [self.get_simulator(self.rnn_one_step_observer.h)]
        X_complete = [x]
        X_res = [x]
        x_shape = x.shape
        for j in range(self.recognition_steps + 1, nSteps):
            control = control_input[:, j : j + 1, :] if control_input is not None else None
            x_observer = (
                self.rnn_one_step_observer(x, control).view(x_shape)
                if self.data_as_input_observer
                else self.rnn_one_step_observer(None, control).view(x_shape)
            )
            if self.rnn_one_step_residuum is not None:
                x_residuum = (
                    self.rnn_one_step_residuum(x, control).view(x_shape)
                    if self.simulator_as_input_residuum
                    else self.rnn_one_step_residuum(x, None).view(x_shape)
                    if self.data_as_input_residuum
                    else self.rnn_one_step_residuum(None, control).view(x_shape)
                )
                X_res.append(x_residuum)
            if self.rnn_one_step_residuum is not None:
                x = x_observer + x_residuum
            else:
                x = x_observer
            X_complete.append(x)
            X_simulator.append(self.get_simulator(self.rnn_one_step_observer.h))
        statelist = [X_simulator, X_complete, X_res]
        for states in range(0, len(statelist)):
            statelist[states] = torch.cat(statelist[states], 1)
        statelist[1] = torch.cat((y_recognition, statelist[1]), 1)
        if control_input is not None:
            statelist[0] = torch.cat((recognition_control, statelist[0]), 1)
        statelist[2] = torch.cat((recognition_control, statelist[2]), 1)
        return statelist[0], statelist[2], statelist[1]


class DampedRollout(RNNRollouts):
    """
    Producing a decaying trajectory with an RNN via initial_damping*exp(damping*-t)*tanh(y)
    y bounded, here via applying tanh() to RNN outputs.
    Args:
        rnn_one_step (RNN): RNN that produces one-step-ahead predictions.
        recognition_steps (int): number of steps for warmup phase
    Attributes:
        rnn_one_step (RNN) (torch.nn.Module): RNN-module that produces one step ahead predictions.
        reg (torch.nn.Tanh()): bounding the outputs.
        inverse_reg (torch.atanh()): inverse of output bounding.
        damping (torch.Tensor): damping factor.
        initial_dampling (torch.Tensor): initial value.
        activation (torch.nn.Softplus): method to make damping factor negative
        recognition_steps (int): number of steps for warmup phase
    """

    def __init__(self, rnn_one_step: Optional[Rollouts] = None, recognition_steps: int = 50, damping=-3.0):
        super().__init__()
        self.rnn_one_step = rnn_one_step if rnn_one_step is not None else GRUModel()
        self.reg = torch.tanh
        self.inverse_reg = torch.atanh
        self.damping = torch.nn.Parameter(torch.Tensor([damping]))
        self.initial_damping = torch.nn.Parameter(torch.Tensor([3.0]))
        self.activation = torch.nn.Softplus()
        self.recognition_steps: int = recognition_steps

    def get_negative_damping(self):
        """
        assure to have a negative damping factor
        """
        return -self.activation(self.damping)

    def damper(self, t):
        """
        computing damping trajectory
        """
        return self.initial_damping * torch.exp(self.get_negative_damping() * t)

    def forward_damping_operation(self, learned_trajectory, t: torch.Tensor):
        """
        compute damping operation by bounding output and multiplying with damper
        """
        learned_output = self.reg(learned_trajectory)
        return learned_output * self.damper(t)

    def inverse_damping_operation(self, original_trajectory, t):
        """
        compute inverse damping operation to extract original signal
        """
        return self.inverse_reg(torch.div(original_trajectory, self.damper(t)))

    def rollout(
        self, y: torch.Tensor, control_input: Optional[torch.Tensor], t: torch.Tensor, nSteps: int
    ) -> Tuple[torch.Tensor, None, None]:
        """
        Args:
            y_recognition ((batchsize, trajectory_length, d_x)-torch.Tensor): training data.
            control_input ((batchsize, trajectory_length, d_u)-torch.Tensor): control input (here: observable states).
        Returns:
            X ((batchsize, recongition_length, d_y)-torch.Tensor): output trajectory.
            None: fit interface.
            None: fit interface.
        """
        damped_trajectory = self.damper(t)
        y_recognition = self.get_recognition_batch(y)
        self.rnn_one_step.initialize()
        recognition_control = control_input[:, 0 : self.recognition_steps, :] if control_input is not None else None
        self.warmup(y_recognition, recognition_control)
        x = y_recognition[:, -1, :].unsqueeze(2)
        X = [x]
        for j in range(self.recognition_steps + 1, nSteps):
            control = control_input[:, j : j + 1, :] if control_input is not None else None
            x = self.rnn_one_step(x, control)
            x = torch.mul(x, damped_trajectory[:, j : j + 1, :])
            X.append(x)
        X = torch.cat(X, 1)
        X = torch.cat((y_recognition, X), 1)
        return X, None, None


class TimeDependentTSModel(TSModel):
    """
    residual model for time series generation.

    Given a simulator trajectory and a warmup trajectory, an observer model (e.g. a KKL-model) is trained to obtain
    observable hidden states. Additionally, these hidden states serve as control input for the recurrent model.
    Thus, the two main components are a hybrid model that receives simulator trajectories as an input and returns
    latent states and projections as an output
    and a recurrent model that gets a short warmup trajectory and optional control inputs as an input and return an
    output prediction.

    Attributes:
        observer_model (torch.nn.Module): hybrid component as KKL-Observer (simulator trajectory as input).
        learning_scheme (torch.nn.Module or None): learning-based model as GRU-model (recognition trajectory and control input).
        in case of None, output is predicted purely based on Observer scheme.
        device (torch.device): device.
        simulator_as_input (bool): indicating, whether the RNN component (learning scheme receives simulator as control
            input or produces a trajectory on its own.
        trainable_simulator (Optional[TrainableSimulator]): Trainable simulator component.
    """

    def __init__(
        self,
        observer_model: Optional[Observer] = None,
        learning_scheme: Optional[Rollouts] = None,
        device: Optional[str] = "cpu",
        simulator_as_input: bool = True,
        trainable_simulator: Optional[TrainableSimulator] = UnmodifiedSimulator(),
        partially_obs_gru: Optional[bool] = False,
    ):
        """
        Args:
            observer_model: hybrid component as KKL-Observer (simulator trajectory as input).
            learning_scheme: learning-based model as GRU-model (recognition trajectory and control input).
            in case of None, output is predicted purely based on Observer scheme.
            device: device.
            simulator_as_input: indicating, whether the RNN component (learning scheme receives simulator as control
            input or produces a trajectory on its own.
            trainable_simulator (Optional[TrainableSimulator]): Trainable Simulator component.
        """
        super(TimeDependentTSModel, self).__init__()
        self.observer_model = observer_model
        self.learning_scheme = learning_scheme
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.simulator_as_input = simulator_as_input
        self.trainable_simulator = trainable_simulator
        self.partially_obs_gru = partially_obs_gru

    def fit(self, batch: list, loss: torch.nn.Module) -> torch.Tensor:
        """

        Args:
            batch (list):
            batch[0]: training data
            batch[1]: simulator data
            loss (torch.nn.Module): loss function.

        Returns:
            loss(batch, output) (torch.Tensor) returning the loss.

        """
        y_n = batch[0]
        s = batch[1] if len(batch) > 1 else None
        t = batch[2] if len(batch) > 2 else None
        simulator_trajectory, learned_trajectory, full_trajectory, reference_s = self.rollout(s, y_n, t, y_n.shape[1])
        return loss(full_trajectory, y_n, simulator_trajectory, reference_s, learned_trajectory)

    def rollout_wrapper(self, test_X: list) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        extracts the necessary data from a list of validation or test data
        Args:
            test_X (list [validation_trajectory, simulator_trajectory]
            with noise-free trajectory: (batchsize, horizon, dim)-torch.Tensor
                 noisy trajectory: (batchsize, horizon, dim)-torch.Tensor
                 simulator_trajectory : (batchsize, horizon, dim)-torch.Tensor
                 time                 : (batchsize, horizon, 1)-torch.Tensor

        Returns:
            list [outputs], here: (batchsize, horizon, dim)-torch.Tensor: simulator trajectory
                                  (batchsize, horizon, dim)-torch.Tensor: learning-based trajectory
                                  (batchsize, horizon, dim)-torch.Tensor: full trajectory

        """
        nSteps = test_X[0].shape[1]
        if len(test_X) > 2:
            test_s = test_X[2]
        else:
            test_s = None
        if len(test_X) > 3:
            test_t = test_X[3]
        else:
            test_t = None

        output = self.rollout(test_s, test_X[1], test_t, nSteps)
        return output

    def get_learning_trajectory(self, s: torch.Tensor, y_n: torch.Tensor, t: torch.Tensor, nSteps: int) -> torch.Tensor:
        """
        computes the RNN-based component either with simulator as control input or without
        Args:
            s (torch.Tensor): simulator trajectory
            y_n (torch.Tensor): training trajectory for warmup
            t (torch.Tensor): training time
        Returns:
            learned_trajectory (torch.Tensor): RNN rollouts
        """
        if self.simulator_as_input:
            learned_trajectory, _, _ = self.learning_scheme.rollout(y_n, s, t, nSteps)
        else:
            learned_trajectory, _, _ = self.learning_scheme.rollout(y_n, None, t, nSteps)
        return learned_trajectory

    def rollout(
        self, s: torch.Tensor, y_n: torch.Tensor, t: torch.Tensor, nSteps: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:

        if self.partially_obs_gru:
            simulator_trajectory, learned_trajectory, full_trajectory, s = self.partially_obs_gru_rollout(
                s, y_n, t, nSteps
            )
        else:
            simulator_trajectory, learned_trajectory, full_trajectory, s = self.standard_rollout(s, y_n, t, nSteps)
        return simulator_trajectory, learned_trajectory, full_trajectory, s

    def standard_rollout(
        self, s: torch.Tensor, y_n: torch.Tensor, t: torch.Tensor, nSteps: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        computes output of residual model (Eq. 11)
        Args:
            s ((batchsize, rollout_length, dim)-torch.Tensor): simulator trajectory.
            y_n ((batchsize, recognition_length, dim)-torch.Tensor): short part of training trajectory for recognition.
            t: ((batchsize, rollout_length, 1)-torch.Tensor): time vector

        Returns:
            simulator_trajectory ((batchsize, rollout_length, simulator_dim)-torch.Tensor)): simulator reconstructions.
            learned_trajectory ((batchsize, rollout_length, dim)-torch.Tensor): learning-based residuum.
            full_trajectory ((batchsize, rollout_length, dim)-torch.Tensor)): output (to reproduce training data).

        """
        if self.trainable_simulator is not None:
            s = self.trainable_simulator(t, s)
        if self.observer_model is not None:
            observer_trajectory, x, reference_s = self.observer_model(s, t)
            simulator_trajectory = self.observer_model.get_simulator_trajectory(x)
        else:
            simulator_trajectory = None
            learned_trajectory = self.get_learning_trajectory(s, y_n, t, nSteps)
            full_trajectory = learned_trajectory
            return simulator_trajectory, learned_trajectory, full_trajectory, s
        if self.learning_scheme is not None:
            learned_trajectory = self.get_learning_trajectory(s, y_n - observer_trajectory[:, :nSteps, :], t, nSteps)
            full_trajectory = learned_trajectory + observer_trajectory
        else:
            full_trajectory = observer_trajectory
            learned_trajectory = None
        return simulator_trajectory, learned_trajectory, full_trajectory, s

    def partially_obs_gru_rollout(
        self, s: torch.Tensor, y_n: torch.Tensor, t: torch.Tensor, nSteps: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        computes output of residual model (Eq. 11)
        Args:
            s ((batchsize, rollout_length, dim)-torch.Tensor): simulator trajectory.
            y_n ((batchsize, recognition_length, dim)-torch.Tensor): short part of training trajectory for recognition.
            t: ((batchsize, rollout_length, 1)-torch.Tensor): time vector

        Returns:
            simulator_trajectory ((batchsize, rollout_length, simulator_dim)-torch.Tensor)): simulator reconstructions.
            learned_trajectory ((batchsize, rollout_length, dim)-torch.Tensor): learning-based residuum.
            full_trajectory ((batchsize, rollout_length, dim)-torch.Tensor)): output (to reproduce training data).

        """
        if self.trainable_simulator is not None:
            s = self.trainable_simulator(t, s)
        if self.simulator_as_input:
            simulator_trajectory, learned_trajectory, full_trajectory = self.learning_scheme.rollout(y_n, s, t, nSteps)
        else:
            simulator_trajectory, learned_trajectory, full_trajectory = self.learning_scheme.rollout(
                y_n, None, t, nSteps
            )
        return simulator_trajectory, learned_trajectory, full_trajectory, s


class FilteredRollouts(RNNRollouts):
    """filtered GRU rollouts

    Args:
        rnn_rollouts (Rollouts): desired rollout scheme
        Wn (torch.double): cutoff frequency.
        btype(str): specifying type of filter
        recognition_steps (int): length of warmup phase
        data_as_input (bool): indicates if data are provided as input

    Attributes:
        numerator, denominator (torch.Tensor, torch.Tensor): filter coefficients
        rnn_rollouts (Rollouts): desired rollout scheme
        recognition_steps (int): length of warmup phase
        data_as_input (bool): indicates if data are provided as input
    """

    def __init__(
        self,
        rnn_rollouts: Rollouts(),
        Wn: torch.double = 0.1,
        btype: str = "highpass",
        recognition_steps: int = 50,
        data_as_input=True,
    ):

        super(RNNRollouts, self).__init__()
        self.numerator, self.denominator = filter_values(Wn, btype)
        self.rnn_rollouts = rnn_rollouts if rnn_rollouts is not None else RNNRollouts()
        self.recognition_steps = recognition_steps
        self.data_as_input = data_as_input

    def rollout(
        self, y: torch.Tensor, control_input: Optional[torch.Tensor], t: torch.Tensor, nSteps: int
    ) -> Tuple[torch.Tensor, None, None]:
        initial_trajectory, _, _ = self.rnn_rollouts.rollout(y, control_input, t, nSteps)
        filtered_trajectory = filter_via_torch(
            self.numerator, self.denominator, initial_trajectory.squeeze(2)
        ).unsqueeze(2)
        return filtered_trajectory, None, None
