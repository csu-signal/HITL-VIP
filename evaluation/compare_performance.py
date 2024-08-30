# USAGE (assumes you're in python_vip folder)
# run batch_file_process.sh to generate analysis_output.txt file(s)
# run python ../evaluation/compare_performance.py -p pilots_analysis_output.txt -t assisted_analysis_output.txt
# (where pilots_analysis_output.txt contains statistics of solo pilot performance and assisted_analysis_output.txt contains statistics of assisted pilot performance)

import argparse
import numpy as np
import pandas as pd
from itertools import groupby

def main():
    parser = argparse.ArgumentParser(
                    prog='compare_performance',
                    description='Did assistants help?')
    parser.add_argument('-p', '--pilots_file', required=True)
    parser.add_argument('-t', '--trials_file', required=False)
    args = parser.parse_args()
        
    pilots_data = open(args.pilots_file,"r").read().split('\n\n')
    pilots_data = [p.split('\n') for p in pilots_data]

    column_heads = ["Number", "Name", "% destabilizing actions", "# crashes", "Distance from DOB mean", "Angular position SD", "Velocity magnitude mean", "Angular velocity SD", "Velocity RMS", "Deflection magnitude mean"]
    
    pilots_rows = []
    for sample in pilots_data:
        if len(sample) > 1:
            row = [s.strip() for s in sample[0].split('Opening')]
            row[1] = row[1].split('/')[-2].split('.')[0]
            row = row + [sample[1].split(':')[-1].replace('%','').strip()]
            row = row + [sample[2].split(':')[-1].strip()]
            row = row + [sample[5].split(':')[-1].strip()]
            row = row + [sample[6].split(':')[-1].strip()]
            row = row + [sample[7].split(':')[-1].strip()]
            row = row + [sample[8].split(':')[-1].strip()]
            row = row + [sample[9].split(':')[-1].strip()]
            row = row + [sample[10].split(':')[-1].strip()]
            pilots_rows.append(row)
    
    pilots_df = pd.DataFrame(pilots_rows, columns = column_heads)
    print(pilots_df)
    
    if args.trials_file is not None:
        trials_data = open(args.trials_file,"r").read().split('\n\n')
        trials_data = [t.split('\n') for t in trials_data]
        
        trials_rows = []
        for sample in trials_data:
            if len(sample) > 1:
                row = [s.strip() for s in sample[0].split('Opening')]
                row[1] = row[1].split('/')[-2].replace('strategy_suggestion_pilot_','').split('.')[0][4:].split('_intermittent')[0]
                row = row + [sample[1].split(':')[-1].replace('%','').strip()]
                row = row + [sample[2].split(':')[-1].strip()]
                row = row + [sample[5].split(':')[-1].strip()]
                row = row + [sample[6].split(':')[-1].strip()]
                row = row + [sample[7].split(':')[-1].strip()]
                row = row + [sample[8].split(':')[-1].strip()]
                row = row + [sample[9].split(':')[-1].strip()]
                row = row + [sample[10].split(':')[-1].strip()]
                trials_rows.append(row)
        
        trials_df = pd.DataFrame(trials_rows, columns = column_heads)
        print(trials_df)
        
        for i in range(len(pilots_df["Name"])):
            trial = np.array(trials_df[trials_df["Name"].str.startswith(pilots_df["Name"][i])].iloc[:,2:],dtype=np.float32)
            if trial.size > 0:
                trial_name = list(trials_df[trials_df["Name"].str.startswith(pilots_df["Name"][i])].iloc[:,1])
                trial_name = [t[0] for t in groupby(trial_name)]
                trial = trial.reshape(26,3,-1)
                trial = np.mean(trial,axis=1)
                trial[:,1] *= 3
                pilot = np.mean(np.array(pilots_df[pilots_df["Name"]==pilots_df["Name"][i]].iloc[:,2:],dtype=np.float32),axis=0)
                pilot[1] *= 3
                print("\nPilot", pilots_df["Name"][i], pilot)
                combined_trial_rows = np.hstack([np.array(trial_name).reshape(-1,1),trial])
                combined_trial_rows = np.vstack([np.hstack([pilots_df["Name"][i],pilot]),combined_trial_rows])
                combined_trial_df = pd.DataFrame(combined_trial_rows, columns = column_heads[1:])
                print("\nCombined trials\n", combined_trial_df)
                diff_rows =  np.hstack([np.array(trial_name).reshape(-1,1),trial-pilot])
                diff_rows = np.vstack([np.hstack([pilots_df["Name"][i],pilot-pilot]),diff_rows])
                diff_df = pd.DataFrame(diff_rows, columns = column_heads[1:])
                print("\nDifference\n", diff_df)
                
                combined_trial_df.to_csv(f"{args.trials_file.split('.')[0]}_combined.csv",index=False)
                diff_df.to_csv(f"{args.trials_file.split('.')[0]}_diff.csv",index=False)
    else:
        pilot_name = np.array(pilots_df)[1::3,1:2]
        pilot_trial = np.array(pilots_df)[:,2:].reshape(26,3,-1)
        pilot_trial = np.array(pilot_trial,dtype=np.float32)
        pilot_trial = np.mean(pilot_trial,axis=1)
        pilot_trial[:,1] *= 3
        combined_trial_rows = np.hstack([np.array(pilot_name).reshape(-1,1),pilot_trial])
        print(combined_trial_rows.shape)
        combined_trial_df = pd.DataFrame(combined_trial_rows, columns = column_heads[1:])
        print("\nCombined trials\n", combined_trial_df)
        
        combined_trial_df.to_csv(f"{args.pilots_file.split('.')[0]}_combined.csv",index=False)

if __name__ == "__main__":
    main()

