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

from hybrid_observer.utils import get_path, create_parentfolder, set_random_seeds, save_validation_data
from hybrid_observer.configs.config_container import get_configs, save_config, save_json_backup
from hybrid_observer.training.data_factory import DataFactory
from hybrid_observer.training.trainer_factory import TrainerFactory
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="process some configurations.")
    parser.add_argument(
        "--training_data_file",
        help="provide the name of training data file",
        default="Exp1.npz",
    )
    parser.add_argument("--experiment_name", help="provide experiment name", default="Experiment 1")
    parser.add_argument("--config_path", help="provide path to json config file", default="Exp1.json")
    parser.add_argument("--random_seed", type=int, help="provide a random seed", default=10)
    return parser


def run_experiment():
    parser = parse_args()
    args = parser.parse_args()
    set_random_seeds(args.random_seed)
    global_config = get_configs(args.config_path)
    #    global_config_json = save_config(global_config, "test.json")
    data_folder = get_path(args.training_data_file)
    parentfolder = create_parentfolder(folder=global_config.folder_enum, experiment_name=args.experiment_name)
    data_loader, test_data = DataFactory.build(global_config.data_config, data_folder)
    trainer = TrainerFactory.build(global_config.training_config, parentfolder)
    #    save_json_backup("Exp1.json", parentfolder)
    save_validation_data(parentfolder, test_data)
    model = trainer.train(test_X=test_data, dataloader=data_loader, random_seed=args.random_seed)
    return model


if __name__ == "__main__":
    run_experiment()
