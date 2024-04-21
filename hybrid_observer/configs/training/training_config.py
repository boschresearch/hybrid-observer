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

from pydantic import BaseSettings, Field

from hybrid_observer.configs.data.data_config import PostprocessorConfig, AvailablePostprocessorConfigs
from hybrid_observer.configs.model.model_config import (
    RolloutConfig,
    RolloutModelsConfig,
    TimeDependentTSModelConfig,
)
from hybrid_observer.configs.training.callback_config import (
    CallbackConfig,
    PlotDataAndSimConfig,
    PlotDataConfig,
    SaveModelConfig,
    ComputeLossesConfig,
    SaveRolloutsConfig,
    PlotLossesConfig,
)
from hybrid_observer.configs.training.enum import LossEnum
from typing import Union, Tuple, List, Any
from hybrid_observer.configs.training.callback_config import FolderManagerConfig


class LossStructureConfig(BaseSettings):
    pass


class StandardLossConfig(LossStructureConfig):
    StandardLossConfig: Any = ""

    class Config:
        arbitrary_types_allowed = True


class DataSimLossConfig(LossStructureConfig):
    DataSimLossConfig: Any = ""
    sim_weight: float = 1


class SimAndDataRegularizedConfig(LossStructureConfig):
    SimAndDataRegularizedConfig: Any = ""
    sim_weight: float = 1
    reg_weight: float = 1


ImplementedLossesConfig = Union[StandardLossConfig, DataSimLossConfig, SimAndDataRegularizedConfig]


class LossConfig(BaseSettings):
    loss_type: LossEnum = LossEnum.RMSE
    loss_structure: ImplementedLossesConfig = Field(default_factory=lambda: DataSimLossConfig())


class AdamConfig(BaseSettings):
    AdamConfig: Any = ""
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0


class SGDConfig(BaseSettings):
    SGDConfig: Any = ""
    lr: float = 1e-3
    momentum: float = 0
    weight_decay: float = 0


ImplementedOptimizers = Union[AdamConfig, SGDConfig]


class StepLRConfig(BaseSettings):
    StepLRConfig: Any = ""
    step_size: int = 100
    gamma: float = 0.02
    last_epoch: int = 1.0


ImplementedSchedulers = Union[StepLRConfig]


class TrainerConfigs(BaseSettings):
    pass


class TrainingConfig(TrainerConfigs):
    TrainingConfig: Any = ""
    model_config: TimeDependentTSModelConfig = Field(default_factory=lambda: TimeDependentTSModelConfig())
    loss_config: LossConfig = Field(default_factory=lambda: LossConfig())
    opt: ImplementedOptimizers = Field(default_factory=lambda: AdamConfig())
    scheduler: ImplementedSchedulers = None
    training_steps: int = 1000
    save_epoch: int = 50
    postprocessor_config: AvailablePostprocessorConfigs = None
    callbacks: List[CallbackConfig] = Field(
        default_factory=lambda: [
            PlotDataAndSimConfig(),
            SaveModelConfig(),
            SaveRolloutsConfig(),
            ComputeLossesConfig(),
            PlotLossesConfig(),
        ]
    )
    folder_manager: FolderManagerConfig = Field(default_factory=lambda: FolderManagerConfig())
    device: str = "cpu"


AvailableTrainerConfigs = Union[TrainingConfig]
