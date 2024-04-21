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

import enum
from enum import Enum


class MetaEnum(enum.EnumMeta):
    def __contains__(cls, item):
        return item in cls.__members__.values()


class LossEnum(str, Enum):
    RMSE = "RMSE"
    RMAE = "RMAE"


class SolverEnum(str, Enum):
    Adam = "Adam"
    SGD = "SGD"


class DataEnum(str, Enum):
    LoadData = "load_data"
    LoadDataWithSim = "load_data_with_sim"
    LoadDataWithSimAndTime = "load_data_with_sim_and_time"


class FolderEnum(Enum):
    NoFolder = 1
    FolderWithTimestamp = 2
    FolderWithoutTimeStamp = 3
