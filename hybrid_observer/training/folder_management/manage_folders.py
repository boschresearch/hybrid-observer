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

from pathlib import Path


class FolderManager:
    def __init__(self, plots_folder: bool = True, model_folder: bool = True, computations_folder: bool = True):
        super(FolderManager, self).__init__()
        self.plots_folder = plots_folder
        self.model_folder = model_folder
        self.computations_folder = computations_folder

    def create_model_folder(self, results_folder):
        if self.model_folder:
            model_folder = Path.joinpath(results_folder, "model")
        else:
            model_folder = results_folder
        if not model_folder.exists():
            model_folder.mkdir(parents=True)
        return model_folder

    def create_plots_folder(self, results_folder):
        if self.plots_folder:
            plots_folder = Path.joinpath(results_folder, "plots")
        else:
            plots_folder = results_folder
        if not plots_folder.exists():
            plots_folder.mkdir(parents=True)
        return plots_folder

    def create_computations_folder(self, results_folder):
        if self.computations_folder:
            computations_folder = Path.joinpath(results_folder, "computations")
        else:
            computations_folder = results_folder
        if not computations_folder.exists():
            computations_folder.mkdir(parents=True)
        return computations_folder

    def get_computations_folder(self, results_folder):
        if self.computations_folder:
            computations_folder = Path.joinpath(results_folder, "computations")
        return computations_folder

    def create_all_folders(self, results_folder):
        model_folder = self.create_model_folder(results_folder)
        computations_folder = self.create_computations_folder(results_folder)
        plots_folder = self.create_plots_folder(results_folder)
        return model_folder, computations_folder, plots_folder
