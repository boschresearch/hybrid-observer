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

# Providing functionalities of config management 

from typing import Optional

from hybrid_observer.configs.global_configuration import GlobalConfig
import shutil
import json
from hybrid_observer.utils import get_path
from typing import Optional
from pathlib import Path


def get_configs(config_name: Optional[str]) -> GlobalConfig:
    """
    creates a config out of string pointing to json file
    """
    if config_name is None:
        global_config = GlobalConfig()
    else:
        config_path = get_path(experiment_name=config_name, type="CONFIG_DIR")
        global_config = GlobalConfig.parse_file(config_path)
    return global_config


def save_config(config: GlobalConfig, config_name: str):
    """
    creates a json config out of config object
    """
    config_path = get_path(experiment_name=config_name, type="CONFIG_DIR")
    with config_path.open("w") as f:
        f.write(config.json())
    return config.json()


def save_json_backup(config_name: str, folder: Path):
    """
    saves json config to predefined folder
    """
    config_path = get_path(experiment_name=config_name, type="CONFIG_DIR")
    new_path = Path.joinpath(folder, config_name)
    shutil.copyfile(config_path, new_path)
