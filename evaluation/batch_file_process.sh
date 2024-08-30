#!/bin/bash
pilots_only=0
assts_only=0
root_dir="../output/"
arrays_file="../evaluation/arrays.txt"

if [[ ! -z "$1" ]]; then
  root_dir=$1
  shift 1
else
  break
fi

if [[ ! -z "$2" ]]; then
  arrays_file=$2
  shift 1
else
  break
fi

if [ "$3" = "--pilots_only" -o "$3" = "-p" ]; then
  pilots_only=1
  shift 1
elif [ "$3" = "--assts_only" -o "$3" = "-a" ]; then
  assts_only=1
  shift 1
else
  break
fi

rm -f analysis_output.txt $arrays_file
touch analysis_output.txt $arrays_file

i=0
FILES="$root_dir/*/pendulum_only_trial_data.csv"

if [ $pilots_only -eq 1 ] ; then
  FILES="$root_dir/pilot_*/pendulum_only_trial_data.csv"
elif [ $pilots_only -eq 1 ] ; then
  FILES="$root_dir/assistant_*/pendulum_only_trial_data.csv"
fi

for f in $FILES
do
  ((i=i+1))
  echo "Processing $i $f..."
  printf "%d " $i >> analysis_output.txt
  python ../evaluation/trial_data_metrics.py -f $f -a $arrays_file >> analysis_output.txt
done
