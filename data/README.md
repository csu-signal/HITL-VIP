## Data file descriptions

The MARS data file can be downloaded from [here](https://figshare.com/s/7d935199c01c0edcafa1). The subset of the VIP data that we used is shared in this repository. 

1. `data_all.csv`: original MARS dataset. Needs to be downloaded from the link provided above. 
2. `data_all_radians.csv`: MARS dataset with Position and Velocity columns converted to radians and radians/seconds for easier integration of deep learning models with PyVIP. This is created using the `mars_data_preprocess.ipynb` notebook mentioned below. 
3. `overall_proficiency.csv`: the overall proficiency displayed by each participant in the MARS trials.
<!-- 4. `VIP/raw_data/`: the raw data for every VIP trial participant.
5. `VIP/data_used/`: the data used for training VIP pilots broken down by proficiency, with only specific trials chosen where the trial setup is analogous to the MARS setup. This can be checked by the `IVPis_protocol-r3_HD.xlsx` protocol file: for the trial to be a MARS analog the `pinc` and `pctcoh` values are 0.02 and 50 respectively. -->
6. `vip_data_all.csv`: the processed VIP data in the same format as MARS data. 
7. `vip_overall_proficiency.csv`: the overall proficiency displayed by the participants whose data was used in training. 

## Scripts

* `mars_data_preprocess.ipynb`: this notebook will preprocess the MARS data by converting the position and velocity features into radians and radians/seconds respectively which is used to train the deep learning models. 
<!-- * `process_vip_data.ipynb` processed the raw data that is to be used into the MARS data file format. -->