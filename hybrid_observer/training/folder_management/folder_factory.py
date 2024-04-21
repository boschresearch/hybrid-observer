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
from hybrid_observer.basic_interfaces import Factory
from hybrid_observer.configs.training.callback_config import FolderManagerConfig
from hybrid_observer.training.folder_management.manage_folders import FolderManager
from pathlib import Path


class FolderFactory(Factory):
    @staticmethod
    def build(config: FolderManagerConfig):
        folder_manager = FolderManager(
            plots_folder=config.plots_folder,
            model_folder=config.model_folder,
            computations_folder=config.computations_folder,
        )
        return folder_manager
