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

from hybrid_observer.model.model_factory import TimeDependentTSModelFactory
from hybrid_observer.training.data_factory import PostprocessorFactory
from hybrid_observer.training.optimizer.optimizers import OptimizerFactory
from hybrid_observer.configs.training.training_config import TrainerConfigs
from hybrid_observer.evaluation.model_evaluation import Evaluate


from hybrid_observer.basic_interfaces import Factory


class EvaluationFactory(Factory):
    @staticmethod
    def build(config: TrainerConfigs) -> Evaluate:
        model = TimeDependentTSModelFactory.build(config.model_config)
        opt = OptimizerFactory.build(config.opt, model)
        postprocessor = (
            PostprocessorFactory.build(config.postprocessor_config) if config.postprocessor_config is not None else None
        )
        return Evaluate(model=model, opt=opt, postprocessor=postprocessor)
