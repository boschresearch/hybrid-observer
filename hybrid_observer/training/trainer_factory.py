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

from typing import Any, Optional

from hybrid_observer.configs.model.model_config import TimeDependentTSModelConfig, RolloutConfig
from hybrid_observer.training.callbacks.callback_factory import CallbackFactory

from hybrid_observer.model.model_factory import TimeDependentTSModelFactory, RolloutFactory
from hybrid_observer.training.data_factory import PostprocessorFactory
from hybrid_observer.training.folder_management.folder_factory import FolderFactory
from hybrid_observer.training.loss.loss_factory import LossFactory
from hybrid_observer.training.optimizer.optimizers import OptimizerFactory, SchedulerFactory
from hybrid_observer.configs.training.training_config import (
    TrainingConfig,
    TrainerConfigs,
)

from hybrid_observer.training.training_components import Trainer

from hybrid_observer.basic_interfaces import Factory
from pathlib import Path


class TrainerFactory:
    @staticmethod
    def build(config: TrainerConfigs, path: Optional[Path]) -> Trainer:
        if config.model_config.__class__ == TimeDependentTSModelConfig:
            model = TimeDependentTSModelFactory.build(config.model_config)
        if config.__class__ == TrainingConfig:
            loss = LossFactory.build(config.loss_config)
            opt = OptimizerFactory.build(config.opt, model)
            scheduler = SchedulerFactory.build(config.scheduler, opt) if config.scheduler is not None else None
            training_steps = config.training_steps
            folder_manager = FolderFactory.build(config.folder_manager)
            model_folder, computations_folder, plots_folder = folder_manager.create_all_folders(path)
            postprocessor = (
                PostprocessorFactory.build(config.postprocessor_config)
                if config.postprocessor_config is not None
                else None
            )
            callbacks = [
                CallbackFactory.build(cfg, model_folder, computations_folder, plots_folder) for cfg in config.callbacks
            ]
            return Trainer(
                model=model,
                callbacks=callbacks,
                loss=loss,
                opt=opt,
                scheduler=scheduler,
                training_steps=training_steps,
                postprocessor=postprocessor,
            )
        else:
            raise NotImplementedError
