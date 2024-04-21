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

from hybrid_observer.configs.training.training_config import (
    TrainingConfig,
    TrainerConfigs,
    AvailableTrainerConfigs,
)
from hybrid_observer.configs.data.data_config import DataConfig
from hybrid_observer.configs.training.enum import FolderEnum
from pydantic import BaseSettings, Field


class GlobalConfig(BaseSettings):
    training_config: AvailableTrainerConfigs = Field(default_factory=lambda: TrainingConfig())
    data_config: DataConfig = Field(default_factory=lambda: DataConfig())
    folder_enum: FolderEnum = FolderEnum(2)
