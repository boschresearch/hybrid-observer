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

#Providing configurations for all models

from typing import Union, Any, Optional

from pydantic import BaseSettings, Field


class ModelBaseConfig(BaseSettings):
    pass


class ObserverConfig(ModelBaseConfig):
    ObserverConfig: Any = ""
    simulator_dim: int = 1
    observation_dim: int = 1
    latent_space_dim: int = 4
    mlp_hidden_dim: int = 100
    device: str = "cpu"


class KKLConfig(ObserverConfig):
    KKLConfig: Any = ""
    mlp_hidden_dim: int = 100
    invertible_dynamics: bool = False


class BasicSimulatorConfig(ObserverConfig):
    BasicSimulatorConfig: Any = ""

    class Config:
        arbitrary_types_allowed = True


ObserverModelsConfig = Union[KKLConfig, BasicSimulatorConfig]


class TrainableSimulatorConfig(ModelBaseConfig):
    TrainableSimulatorConfig: Any = ""

    class Config:
        arbitrary_types_allowed = True


class TrainableSineSimulatorConfig(TrainableSimulatorConfig):
    TrainableSineSimulatorConfig: Any = ""
    amplitude: float = float(1.0)
    frequency: float = float(0.007)
    frequency_shift: float = float(0.0)


StandardSimulatorConfig = Union[TrainableSineSimulatorConfig]


class FilteredSimulatorConfig(TrainableSimulatorConfig):
    FilteredSimulatorConfig: Any = ""
    Wn: float = 0.2
    btype: str = "lowpass"
    trainable_simulator_config: StandardSimulatorConfig = Field(default_factory=lambda: TrainableSineSimulatorConfig())


class UnmodifiedSimulatorConfig(TrainableSimulatorConfig):
    UnmodifiedSimulatorConfig: Any = ""

    class Config:
        arbitrary_types_allowed = True


TrainableSimulatorModelsConfig = Union[TrainableSineSimulatorConfig, UnmodifiedSimulatorConfig, FilteredSimulatorConfig]


class RNNConfig(ModelBaseConfig):
    RNNConfig: Any = ""
    input_dim: int = 1
    output_dim: int = 1
    device: str = "cpu"


class GRUConfig(RNNConfig):
    GRUConfig: Any = ""
    input_dim: int = 1
    output_dim: int = 1
    hidden_dim: int = 64
    device: str = "cpu"


RNNModelsConfig = Union[GRUConfig]


class RolloutConfig(ModelBaseConfig):
    RolloutConfig: Any = ""
    RNN_config: RNNModelsConfig = Field(default_factory=lambda: GRUConfig())
    recognition_steps: int = 50
    time_dependence: bool = False
    data_as_input: bool = True


class DampedRolloutsConfig(ModelBaseConfig):
    DampedRolloutsConfig: Any = ""
    RNN_config: RNNModelsConfig = Field(default_factory=lambda: GRUConfig())
    recognition_steps: int = 50
    time_dependence: bool = False
    damping: float = -3.0


class SummedRNNRolloutsConfig(ModelBaseConfig):
    SummedRNNRolloutsConfig: Any = ""
    rnn_one_step_observer_config: RNNModelsConfig = Field(default_factory=lambda: GRUConfig())
    rnn_one_step_residuum_config: Optional[RNNModelsConfig] = Field(default_factory=lambda: GRUConfig())
    data_as_input_observer: bool = True
    data_as_input_residuum: bool = True
    simulator_dim: int = 1
    recognition_steps: int = 50
    simulator_as_input_residuum: bool = True


PureRolloutsConfig = Union[RolloutConfig, DampedRolloutsConfig]


class FilteredRolloutsConfig(ModelBaseConfig):
    FilteredRolloutsConfig: Any = ""
    rnn_rollouts_config: PureRolloutsConfig = Field(default_factory=lambda: RolloutConfig())
    Wn: float = 0.2
    btype: str = "highpass"
    recognition_steps: int = 50
    data_as_input: bool = True


RolloutModelsConfig = Union[RolloutConfig, DampedRolloutsConfig, FilteredRolloutsConfig, SummedRNNRolloutsConfig]


class TimeDependentTSModelConfig(ModelBaseConfig):
    TimeDependentTSModelConfig: Any = ""
    observer_config: Optional[Union[ObserverModelsConfig]] = None
    learning_config: Optional[RolloutModelsConfig] = SummedRNNRolloutsConfig()
    trainable_simulator_config: Optional[TrainableSimulatorModelsConfig] = Field(
        default_factory=lambda: UnmodifiedSimulatorConfig()
    )
    device: str = "cpu"
    simulator_as_input: bool = False
    time_dependence: bool = True
    partially_obs_gru: bool = False


ModelConfig = Union[ObserverConfig, KKLConfig, RNNConfig, GRUConfig, RolloutConfig]
