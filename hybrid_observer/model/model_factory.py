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

from typing import Any, Union

from hybrid_observer.model.available_models import (
    KKL,
    GRUModel,
    BasicSimulator,
    DampedRollout,
    RNNRollouts,
    TimeDependentTSModel,
    TrainableSimulator,
    TrainableSineSimulator,
    UnmodifiedSimulator,
    FilteredRollouts,
    FilteredSimulator,
    SummedRNNRollouts,
)

from hybrid_observer.model.interfaces import Observer, RNN, Rollouts, TSModel


from hybrid_observer.configs.model.model_config import (
    ModelConfig,
    KKLConfig,
    ObserverConfig,
    RNNConfig,
    GRUConfig,
    BasicSimulatorConfig,
    RolloutConfig,
    DampedRolloutsConfig,
    TimeDependentTSModelConfig,
    RNNModelsConfig,
    RolloutModelsConfig,
    TrainableSimulatorConfig,
    TrainableSineSimulatorConfig,
    UnmodifiedSimulatorConfig,
    FilteredRolloutsConfig,
    FilteredSimulatorConfig,
    SummedRNNRolloutsConfig,
)

from hybrid_observer.basic_interfaces import Factory


class ObserverFactory(Factory):
    @staticmethod
    def build(config: ObserverConfig) -> Observer:
        if config.__class__ == KKLConfig:
            return KKL(
                simulator_dim=config.simulator_dim,
                observation_dim=config.observation_dim,
                latent_space_dim=config.latent_space_dim,
                mlp_hidden_dim=config.mlp_hidden_dim,
                invertible_dynamics=config.invertible_dynamics,
                device=config.device,
            )
        elif config.__class__ == BasicSimulatorConfig:
            return BasicSimulator()
        else:
            raise NotImplementedError


class RNNFactory(Factory):
    @staticmethod
    def build(config: RNNModelsConfig) -> RNN:
        if config.__class__ == GRUConfig:
            return GRUModel(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.output_dim,
                device=config.device,
            )
        else:
            raise NotImplementedError


class RolloutFactory(Factory):
    @staticmethod
    def build(config: RolloutModelsConfig) -> Rollouts:
        if config.__class__ == RolloutConfig:
            return RNNRollouts(
                rnn_one_step=RNNFactory.build(config.RNN_config),
                recognition_steps=config.recognition_steps,
                data_as_input=config.data_as_input,
            )
        elif config.__class__ == DampedRolloutsConfig:
            return DampedRollout(
                rnn_one_step=RNNFactory.build(config.RNN_config),
                recognition_steps=config.recognition_steps,
                damping=config.damping,
            )
        elif config.__class__ == FilteredRolloutsConfig:
            return FilteredRollouts(
                rnn_rollouts=RolloutFactory.build(config.rnn_rollouts_config),
                Wn=config.Wn,
                btype=config.btype,
                recognition_steps=config.recognition_steps,
                data_as_input=config.data_as_input,
            )
        elif config.__class__ == SummedRNNRolloutsConfig:
            return SummedRNNRollouts(
                rnn_one_step_observer=RNNFactory.build(config.rnn_one_step_observer_config),
                rnn_one_step_residuum=RNNFactory.build(config.rnn_one_step_residuum_config)
                if config.rnn_one_step_residuum_config is not None
                else None,
                data_as_input_observer=config.data_as_input_observer,
                data_as_input_residuum=config.data_as_input_residuum,
                simulator_dim=config.simulator_dim,
                recognition_steps=config.recognition_steps,
                simulator_as_input_residuum=config.simulator_as_input_residuum,
            )
        else:
            raise NotImplementedError


class TimeDependentTSModelFactory(Factory):
    @staticmethod
    def build(config: TimeDependentTSModelConfig) -> TSModel:
        return TimeDependentTSModel(
            observer_model=ObserverFactory.build(config.observer_config)
            if config.observer_config is not None
            else None,
            learning_scheme=RolloutFactory.build(config.learning_config)
            if config.learning_config is not None
            else None,
            device=config.device,
            simulator_as_input=config.simulator_as_input,
            trainable_simulator=TrainableSimulatorFactory.build(config.trainable_simulator_config)
            if config.trainable_simulator_config is not None
            else None,
            partially_obs_gru=config.partially_obs_gru,
        )


class TrainableSimulatorFactory(Factory):
    @staticmethod
    def build(config: TrainableSimulatorConfig):
        if config.__class__ == TrainableSineSimulatorConfig:
            return TrainableSineSimulator(
                amplitude=config.amplitude, frequency=config.frequency, frequency_shift=config.frequency_shift
            )
        elif config.__class__ == UnmodifiedSimulatorConfig:
            return UnmodifiedSimulator()
        elif config.__class__ == FilteredSimulatorConfig:
            return FilteredSimulator(
                Wn=config.Wn,
                btype=config.btype,
                trainable_simulator=TrainableSimulatorFactory.build(config.trainable_simulator_config),
            )
        else:
            raise NotImplementedError
