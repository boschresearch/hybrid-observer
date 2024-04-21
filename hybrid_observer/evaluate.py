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

from hybrid_observer.evaluation.model_evaluation import Evaluate
from hybrid_observer.training.data_factory import DataFactory
from hybrid_observer.utils import get_path, create_parentfolder
from hybrid_observer.configs.config_container import get_configs, save_config
from hybrid_observer.evaluation.evaluation_factory import EvaluationFactory
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="process some configurations.")
    parser.add_argument(
        "--training_data_file",
        help="provide the name of training data file",
        default="Exp1.npz",
    )
    parser.add_argument("--experiment_name", help="provide experiment name", default="Experiment 1")
    parser.add_argument("--config_path", help="provide path to json config file", default="Exp1_KKLRNN.json")
    parser.add_argument(
        "--path",
        help="provide the path to the saved model",
        default="Exp1_KKLRNN.pt",
    )
    return parser


def evaluate_model():
    parser = parse_args()
    args = parser.parse_args()
    global_config = get_configs(args.config_path)
    data_folder = get_path(args.training_data_file)
    model_folder = get_path(args.path, "RESULTS_DIR")
    _, validation_data = DataFactory.build(global_config.data_config, data_folder)
    evaluator: Evaluate = EvaluationFactory.build(global_config.training_config)
    model, _, _ = evaluator.get_model(model_folder)
    output = evaluator.rollout_model(model, validation_data)
    evaluator.plot_evaluation(validation_data, output)
    RMSE = evaluator.compute_RMSE(validation_data, output)
    print("RMSE: " + str(RMSE))


if __name__ == "__main__":
    evaluate_model()
