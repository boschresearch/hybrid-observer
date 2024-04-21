Will we give an overview of the components and functionalities of the framework and which configurations will lead to which setting.

# Getting started
We provide some steps to set up a new project.
## Defining your folders
   - define via the [.env template](hybrid_observer/template.env), where your data, configs and results are stored.  
## Create your data
   - save the data in an .npz file. Use folder you defined in .env. 
   - An example for data generation can be seen [here](hybrid_observer/datasets/generate_mass_spring_data.py).
## Configure your data:
   - provide a configuration in a json file or use the default configuration. Store the json file in the folder defined in .env.
## Train your method:
   - An example can be seen in [here](hybrid_observer/training/training_script.py).
   - Hint: if you do not parse a config, it will use the default configuration [GlobalConfiguration](hybrid_observer/configs/global_configuration.py).
## Evaluate your trained model
   - use [evaluate.py](hybrid_observer/evaluate.py) as an example how to evaluate your model.
## Check out your results
   - Depending on your folder configuration and your callbacks, you have your desired results now. 
   - Check them out in the results folder defined in the [.env template](hybrid_observer/template.env).
# Getting your custom configuration
We will first give an overview of the main components and functionalities and then point to example configurations.

# Models
- The framework heavily relies on the interfaces defined. We will give an overview over the interfaces.
- New methods/ models can be inserted by respecting these interfaces. 
## TSModel
   - Interface [TSModel](hybrid_observer/interfaces.py) fits [training method](Trainer.train).
   - consisting of fit method, rollout method and rollout_wrapper method.
- If you want to write a new TSModel:
   - respect interface TSModel.
   - write [ModelBaseConfig](hybrid_observer/configs/model/model_config.py).
   - register model
   - update [TimeDependentTSModelFactory](hybrid_observer/model/model_factory.py).
## TimeDependentTSModel
   - The core of each standard scenario is a [TimeDependendentTSModel](hybrid_observer/project_skeleton/hybrid_observer/model/model_factory.py) and the corresponding [TimeDependendentTSModelConfig](hybrid_observer/configs/model/model_config.py). By changing its components, you get your model.
   - mathematically, it sums the two trajectories of the Observer and Rollout model. It also contains a trainable simulator that is overwritten in case fixed simulator data are given. 
   
## Losses
   - see [Loss](hybrid_observer/project_skeleton/hybrid_observer/training/loss/losses.py). Each loss consists of a [basic_loss](hybrid_observer/training/loss/losses.py) and a Loss Configuration that determines how the loss is computed, configured in loss_type. Addionally, there are three options for loss_structure.
   - [StandardLoss](hybrid_observer/hybrid_observer/training/loss/losses.py): Computes loss between training data and reference (only on target trajectory, not on simulator).
   - DataSimLoss: Computes basic_loss between observations, training results and simulator reference and trained simulator basic_loss(Data,Predictions)+basic_loss(Simulator,Simulator Predictions).
   - RegularizedDataSimLoss: as DataSimLoss but penalizing the purely learning-based influence
   - available basic losses: RMSE and RMAE stored as [LossEnum](hybrid_observer/configs/training/enum.py).
   
## Training
   - The class [Trainer](hybrid_observer/training/training_components.py) takes care of training.
   - Main method: [Trainer.train](training model).
   
## Evaluation
   - The class [Evaluate](hybrid_observer/evaluation/model_evaluation.py) takes care of evaluation.
   
## Data Management
   - Depending on the type of data, either only data, data and simulator or data, simulator and time are loaded.
   - Management of the class is provided in [TrainTest](hybrid_observer/training/training_components.py).
   - The corresponding configuration is provided in [DataConfig](hybrid_observer/configs/data/data_config.py).

## Further Components
   - Callbacks: Controls Training (saves models, losses etc. at checkpoints)
   - Optimizers: Manage type of optimizer.
   - Scheduler: Manage scheduling of optimizer. 
   - FolderManagement: Manage, where results, models etc. are stored. 
   

# Reproducing experiments
 
## Configs: 
To train a model, run the training_script.py providing the data as string, an experiment name, the name of the config and a random seed. 
- train via command line with
```json
training_script.py --training_data_file "name_of_training_data" --experiment_name "name of exp" --config_path "name of config" --random_seed some_int
```
- In the following we list the configs and data for each experiment. In the data folder, we provide configs named ExpNumber_method.json and data ExpNumber.npz or ExpNumber_Filter.npz
  - Damped oscillations (Exp1): 
    - data:
    ```json
    --training_data_file "Exp1.npz" or for filter: "Exp1_Filter.npz
    ```
    - configs: 
    ```json
    --config_path "..."
    ```

      - KKL-RNN: "Exp1_KKLRNN.json" 
      - Hybrid_GRU: "Exp1_hybrid_GRU.json"
      - GRU: "Exp1_GRU.json"  
      - Residual model: "Exp1_Resmodel.json"
      - Filter: "Exp1_Filter.json"
      - invertible transformation KKL-RNN: "Exp1_invertibleKKLRNN.json" 
  - Double-torsion (Exp2):
    - data: 
    ```json
       --training_data_file "Exp2.npz" or for filter: "Exp2_Filter.npz
    ```
    - configs 
    ```json
    --config_path "..."
    ```
      - KKL-RNN: "Exp2_KKLRNN.json" 
      - Hybrid_GRU: "Exp2_hybrid_GRU.json"
      - GRU: "Exp2_GRU.json"  
      - Residual model: "Exp2_Resmodel.json"
      - Filter: "Exp2_Filter.json"
  - Drill-string system (Exp3):
    - data: 
    ```json
       --training_data_file "Exp2.npz" or for filter: "Exp2_Filter.npz
    ```
    - configs:
    ```json
       --config_path "..."
    ```
      - KKL-RNN: "Exp3_KKLRNN.json" 
      - Hybrid_GRU: "Exp3_hybrid_GRU.json"
      - GRU: "Exp3_GRU.json"  
      - Residual model: "Exp3_Resmodel.json"
      - Filter: "Exp3_Filter.json"
  - Van-der-Pol oscillator (Exp4):
    - data: 
    ```json
       --training_data_file "Exp4.npz"
    ```
      - configs:
    ```json
      --config_path "..."
    ```
      - KKL-RNN: "Exp3_KKLRNN.json" 
      - Obs_GRU: "Exp3_ObsGRU.json"
      - GRU: "Exp3_GRU.json"  
      - Residual model: "Exp3_Resmodel.json"
      
## Generate_data: 
For all experiments, data are available. To re-generate them, run the scripts [here](hybrid_observer/datasets).        

## Evaluate pretrained models: 
- For our KKL-RNN, we provide pretrained models in the results folder. 
- Run the evaluation script via
```json
   evaulate.py --training_data_file "name of training_data" --experiment_name "name_of_exp" --config_path "name of config" --path "path to model" 
```
- The first three arguments are the same as for training. 
- The path points to a subfolder of the results folder defined in .env. We provide the following models:
  - Exp1_KKLRNN.pt
  - Exp2_KKLRNN.pt
  - Exp3_KKLRNN.pt
  - Exp4_KKLRNN.pt 