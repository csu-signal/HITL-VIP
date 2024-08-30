# README

## Scripts

* `batch_file_process.sh`: Process all files in a directory into summaries using `trial_data_metrics.py`.  USAGE: `sh batch_file_process.sh <root dir> <output "arrays" file>` \[Optional: `--pilots_only` OR `--assts_only` will only process files of this type and exclude co-performance trials\].
* `evaluate_all_assistants.sh`: Evaluate all assistant models (denoted as all those in the `/working_models/assistants/` folder) running the task alone. USAGE: `sh evaluate_all_assistants.sh`.
* `evaluate_all_pilots.sh`: Evaluate all pilot models (denoted as all those in the `/working_models/pilots/` folder) running the task alone. USAGE: `sh evaluate_all_pilots.sh`.
* `run_bad_evals.sh`: Run all evaluations with "bad" pilots (all config files matching `/configs/simulation/bad_*.json/`). USAGE: `sh run_bad_evals.sh`.
* `run_good_evals.sh`: Run all evaluations with "good" pilots (all config files matching `/configs/simulation/good_*.json/`). USAGE: `sh run_good_evals.sh`.
* `run_med_evals.sh`: Run all evaluations with "medium" pilots (all config files matching `/configs/simulation/med_*.json/`). USAGE: `sh run_med_evals.sh`.

## Python files

* `clusters.py`: Runs k-means clustering over pilot performance to determine "good," "medium," and "bad" exemplars (can run over solo pilot statistics or assisted pilot statistics).
* `compare_performance.py`: Compares the performance of solo pilots with the same pilots when assisted by various assistant models.
* `make_configs.py`: Automatically generates experimental configurations from `evals.csv` (table of pilot/assistant combos). `toy_evals.csv` can be used to create configs for only the models that are included in the submission (not all models could be included for space reasons).
* `trial_data_metrics.py`: Processes individual trial `pendulum_only_trial_data.csv` files into a .txt summary of relevant features (not recommended to run standalone, as this is handled more efficiently through `batch_file_process.sh`).
* `analysis_trends_plot.ipynb` creates plots of the diff output files from `compare_performance.py`. 

Usage information for each is given in comments at the top of each file.
