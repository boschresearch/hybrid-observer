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

# Providing the configurations for data 

from pydantic import BaseSettings, Field
import torch
from pathlib import Path
from hybrid_observer.configs.training.enum import DataEnum
from typing import Optional, Union, Any


class PreprocessorConfig(BaseSettings):
    pass


class FilterConfig(PreprocessorConfig):
    FilterConfig: Any = ""
    order: int = 1
    cutoff: float = 0.07
    mode: str = "lowpass"
    type: str = "butter"
    downsampling_rate: int = 2


AvailablePreprocessorConfigs = Union[FilterConfig]


class PostprocessorConfig(BaseSettings):
    pass


class UpsamplerConfig(PostprocessorConfig):
    UpsamplerConfig: Any = ""
    upsampling_rate: int = 2


AvailablePostprocessorConfigs = Union[UpsamplerConfig]


class TrainTestConfig(BaseSettings):
    training_split: int = 500
    validation_ind: int = 0
    remove_n_train_data: Optional[int] = None
    remove_n_eval_data: Optional[int] = None
    n_dimensional: bool = False
    preprocessor: Optional[AvailablePreprocessorConfigs] = None
    postprocessor: Optional[AvailablePostprocessorConfigs] = None


class DataLoaderConfig(BaseSettings):
    batchsize: int = 50
    isshuffle: bool = True
    length: int = 100
    device: str = "cpu"


class DataConfig(BaseSettings):
    device: str = "cpu"
    train_test_config: TrainTestConfig = Field(default_factory=lambda: TrainTestConfig())
    data_loader_config: DataLoaderConfig = Field(default_factory=lambda: DataLoaderConfig())
    data_type: DataEnum = DataEnum.LoadDataWithSim
