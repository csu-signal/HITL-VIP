#!/bin/bash
FILES="../configs/simulation/bad_*.json"

i=0
while [ $i -lt 3 ]
do
  for f in $FILES
  do
    echo "Running config $f..."
    python vip.py --experiment_config $f --eval_mode
  done
  ((i=i+1))
done
