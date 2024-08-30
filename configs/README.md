## README

This directory contains the different configs for:

1. `training/`: train deep learning or reinforcement learning models.
2. `simulation/`: run the PyVIP simulations with trained models.


### Simulation


To create the configs perform the following steps:

    
    cd python_vip
    python ../evaluation/make_configs.py -f ../evaluation/toy_evals.csv -o ../configs/simulation
    

### Training
The training configs for the subset of models submitted have been provided. To run them follow the steps provided in the `deep_learning/` directory. 