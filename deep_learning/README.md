## How to run the experiments

### Requirements

The following python (3.9.16) libraries are used and can be installed using pip:
  
  `pytorch-lightning torch torchmetrics numpy datasets gluonts orjson evaluate accelerate ujson transformers==4.31.0 tqdm`

### Config file

In the `../configs/training/` directory, there is an example config that defines what a training config will look like.


1. In each training config there is a `dl_training_parameters` dictionary which defines the parameters for training a deep learning model.

### Training the models

Once a config file has been created, training of the models can begin.


Before running the scripts, make sure to update the `data_path` and `config_path` variables (if present) to reflect where the data and training config files are present.

To train and test use the following scripts:

1. `python train.py <train_config_path> <gpu_num>` 
    * `gpu_num` is an optional parameter which could be 0 or 1 or 2 to tell which gpu to use depending on how many gpus are present on the machine. The training process will only use a single GPU. If no GPU is detected then it will run on CPU by default. 
2. `python test.py <train_config_path>`
    * this script tests the trained model against leftout trials. 


#### Informer transformer
To train an instance of the Informer model, you can use the `train_transformer.py` script or run the `train_transformer.ipynb` notebook that uses a toy dataset created from 2 trials of the MARS data. 
The training config file is available at `../configs/training/pilot_mars_informer_good_small_window_future.json`
#### Output

By default, the saved checkpoints are saved in an `output/` directory within `deep_learning`. The checkpoints would named as:
* `output/<name>/best-checkpoint.ckpt` for the MLP, RNN, LSTM, and GRU networks.
    * copy the checkpoint using `cp output/<name>/best-checkpoint.ckpt ../working_models/<type>/<name>.ckpt`
* `output/<name>/Informer-Transformer-roll-<rolling_window_size>-window-<window_size>-future-<prediction_length>/` for the Informer model. 
    * copy the checkpoint using `cp output/<name>/best-checkpoint.ckpt ../working_models/<type>/<name>.ckpt`


### Known issues

* On Apple silicon the dataloader in `train.py` does not work with num_workers argument passed. Comment those for it to work. 