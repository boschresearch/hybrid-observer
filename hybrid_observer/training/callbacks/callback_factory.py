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
from typing import Any
from hybrid_observer.configs.training.callback_config import FolderManagerConfig
from pathlib import Path
from hybrid_observer.training.folder_management.folder_factory import FolderFactory

from hybrid_observer.training.callbacks.available_callbacks import (
    Callbacks,
    PlotDataAndSim,
    PlotData,
    SaveModel,
    SaveRollouts,
    ComputeLosses,
    PlotLosses,
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
from pydantic import BaseSettings

from hybrid_observer.basic_interfaces import Factory


class CallbackFactory:
    @staticmethod
    def build(config: CallbackConfig, model_folder: Path, computations_folder: Path, plots_folder: Path) -> Callbacks:
        if isinstance(config, PlotDataAndSimConfig):
            return PlotDataAndSim(callback_epoch=config.callback_epoch, folder=plots_folder, show_plot=config.show_plot)
        elif isinstance(config, PlotDataConfig):
            return PlotData(callback_epoch=config.callback_epoch, folder=plots_folder, show_plot=config.show_plot)
        elif isinstance(config, ComputeLossesConfig):
            return ComputeLosses(callback_epoch=config.callback_epoch, folder=computations_folder)
        elif isinstance(config, SaveModelConfig):
            return SaveModel(callback_epoch=config.callback_epoch, folder=model_folder)
        elif isinstance(config, SaveRolloutsConfig):
            return SaveRollouts(callback_epoch=config.callback_epoch, folder=computations_folder)
        elif isinstance(config, PlotLossesConfig):
            return PlotLosses(callback_epoch=config.callback_epoch, folder=plots_folder, show_plot=config.show_plot)
        else:
            raise NotImplementedError
