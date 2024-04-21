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

# Providing configurations for callbacks 

from typing import Union, Any

from pydantic import BaseSettings, Field, BaseSettings


class FolderManagerConfig(BaseSettings):
    FolderManagerConfig: Any = ""
    plots_folder: bool = True
    model_folder: bool = True
    computations_folder: bool = True


class PlotDataAndSimConfig(BaseSettings):
    PlotDataAndSimConfig: Any = ""
    callback_epoch: int = 50
    show_plot: bool = True


class PlotDataConfig(BaseSettings):
    PlotDataConfig: Any = ""
    callback_epoch: int = 50
    show_plot: bool = True


class SaveModelConfig(BaseSettings):
    SaveModelConfig: Any = ""
    callback_epoch: int = 50


class ComputeLossesConfig(BaseSettings):
    ComputeLossesConfig: Any = ""
    callback_epoch: int = 50


class SaveRolloutsConfig(BaseSettings):
    SaveRolloutsConfig: Any = ""
    callback_epoch: int = 50


class PlotLossesConfig(BaseSettings):
    PlotLossesConfig: Any = ""
    callback_epoch: int = 50
    show_plot: bool = True


CallbackConfig = Union[
    PlotDataAndSimConfig, PlotDataConfig, SaveModelConfig, ComputeLossesConfig, SaveRolloutsConfig, PlotLossesConfig
]
