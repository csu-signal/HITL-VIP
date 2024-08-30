# USAGE (assumes you're in python_vip folder)
# run python ../evaluation/trial_data_metrics.py -f <path to a pendulum_only_trial_data.csv file> -a ../evaluation/arrays.txt
# Not recommended to run alone: batch_file_process.sh runs this for a directory full of files

import argparse
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
                    prog='trial_data_metrics',
                    description='Analysis of PyVIP trial')
    parser.add_argument('-f', '--filename', required=True)
    parser.add_argument('-a', '--arrays', required=False)
    args = parser.parse_args()
    print(f"Opening {args.filename}")
    
    df = pd.read_csv(args.filename)
    
    no_assistant = False
    
    if not args.filename.split('/')[-2].startswith("assistant") and \
        not args.filename.split('/')[-2].startswith("sac") and \
        not args.filename.split('/')[-2].startswith("ddpg"):
        assert "_good" in args.filename or "_med" in args.filename or "_bad" in args.filename
    proficiency = 4 if "_all_prof" in args.filename else 3 if "_good_med" in args.filename else 2 if "_good" in args.filename else 1 if "_med" in args.filename else 0
            
    num_actions = np.size(df["destabilizing_actions"])
    num_destabilizing_actions = np.sum(df["destabilizing_actions"])
    destabilizing_prop = num_destabilizing_actions/num_actions
    
    num_crashes = np.sum(np.array(df["action_made_by"]) == 0.0)-1
    
    last_timestamp = np.array(df["time"])[-1]
    crash_freq = num_crashes/last_timestamp
        
    avg_crash_probability = np.mean(np.array(df["crash_probabilities"]))
    
    avg_dob_dist = np.mean(np.abs(np.array(df["angular position"])))
    
    sd_angular_pos = np.std(np.array(df["angular position"]))
    
    avg_angvel_mag = np.mean(np.abs(np.array(df["angular velocity"])))
    
    sd_angvel = np.std(np.array(df["angular velocity"]))
    
    angvel_rms = np.sqrt(np.mean(np.array(df["angular velocity"])**2))
    
    avg_defl_mag = np.mean(np.abs(np.array(df["joystick deflection"])))
    
    if np.mean(df["assistant_actions"] == 0.0):
        no_assistant = True
        
    if not no_assistant:
        pilot_asst_diff_mean = np.mean(np.abs(np.array(df["pilot_actions"])-np.array(df["assistant_actions"])))
        pilot_asst_diff_sd = np.std(np.abs(np.array(df["pilot_actions"])-np.array(df["assistant_actions"])))
    
    print(f"% destabilizing actions: {destabilizing_prop*100:.02f}%")
    print(f"# crashes: {num_crashes}")
    print(f"Crash freq: {crash_freq:.04} Hz")
    print(f"Crash prob. mean: {avg_crash_probability*100:.02f}%")
    print(f"Distance from DOB mean: {avg_dob_dist:.02f}")
    print(f"Angular position SD: {sd_angular_pos:.02f}")
    print(f"Velocity magnitude mean: {avg_angvel_mag:.02f}")
    print(f"Angular velocity SD: {sd_angvel:.02f}")
    print(f"Velocity RMS: {angvel_rms:.02f}")
    print(f"Deflection magnitude mean: {avg_defl_mag:.04f}")
    
    if not no_assistant:
        print(f"Pilot-assistant difference mean: {pilot_asst_diff_mean:.04f}")
        print(f"Pilot-assistant difference SD: {pilot_asst_diff_sd:.04f}")

    print()
    
    if args.arrays is not None:
        arrays_file = open(args.arrays, "a")
        arrays_file.write(f"{proficiency} {crash_freq} {destabilizing_prop} {avg_dob_dist} {sd_angular_pos} {avg_angvel_mag} {sd_angvel} {angvel_rms} {avg_defl_mag}\n")
        arrays_file.close()
    
if __name__ == "__main__":
    main()
