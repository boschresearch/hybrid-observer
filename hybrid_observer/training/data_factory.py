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

from hybrid_observer.training.training_components import Dataloader, TrainTest
from hybrid_observer.configs.data.data_config import (
    DataLoaderConfig,
    DataConfig,
    TrainTestConfig,
    AvailablePreprocessorConfigs,
    FilterConfig,
    UpsamplerConfig,
)
from hybrid_observer.basic_interfaces import Factory
from hybrid_observer.configs.training.enum import DataEnum
from pathlib import Path
from hybrid_observer.training.preprocessing import Filter, Upsampler


class PreprocessorFactory(Factory):
    @staticmethod
    def build(config: AvailablePreprocessorConfigs):
        if config.__class__ == FilterConfig:
            return Filter(
                order=config.order,
                cutoff=config.cutoff,
                mode=config.mode,
                type=config.type,
                downsampling_rate=config.downsampling_rate,
            )
        else:
            raise NotImplementedError


class PostprocessorFactory(Factory):
    @staticmethod
    def build(config: AvailablePreprocessorConfigs):
        if config.__class__ == UpsamplerConfig:
            return Upsampler(upsampling_rate=config.upsampling_rate)
        else:
            raise NotImplementedError


class DataloaderFactory(Factory):
    @staticmethod
    def build(config: DataLoaderConfig) -> Dataloader:
        return Dataloader(
            batchsize=config.batchsize, isshuffle=config.isshuffle, length=config.length, device=config.device
        )


class TrainTestFactory(Factory):
    @staticmethod
    def build(config: TrainTestConfig):
        return TrainTest(
            training_split=config.training_split,
            validation_ind=config.validation_ind,
            remove_n_train_data=config.remove_n_train_data,
            remove_n_eval_data=config.remove_n_eval_data,
            n_dimensional=config.n_dimensional,
            preprocessor=PreprocessorFactory.build(config.preprocessor) if config.preprocessor is not None else None,
        )


class DataFactory(Factory):
    @staticmethod
    def build(config: DataConfig, path: Path):
        train_test = TrainTestFactory.build(config.train_test_config)
        dataloader = DataloaderFactory.build(config.data_loader_config)
        if config.data_type == DataEnum.LoadData:
            data, test_data = train_test.load_data(path)
        elif config.data_type == DataEnum.LoadDataWithSim:
            data, test_data = train_test.load_data_with_sim(path)
        elif config.data_type == DataEnum.LoadDataWithSimAndTime:
            data, test_data = train_test.load_data_with_sim_and_time(path)
        else:
            raise NotImplementedError
        data_loader = dataloader.get_dataloader(data)
        return data_loader, test_data
