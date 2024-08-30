#!/bin/bash
FILES="../working_models/pilots/*"

for f in $FILES
do
  t="mlp"
  w="0.5"
  if [[ $f == *"mlp"* ]]; then
    t="mlp"
  elif [[ $f == *"rnn"* ]]; then
    t="rnn"
  elif [[ $f == *"lstm"* ]]; then
    t="lstm"
  elif [[ $f == *"gru"* ]]; then
    t="gru"
  elif [[ $f == *"informer"* ]]; then
    t="informer"
  elif [[ $f == *"sac"* ]]; then
    t="sac"
  elif [[ $f == *"ddpg"* ]]; then
    t="ddpg"
  fi

  if [[ $f == *"future"* ]]; then
    w="0.3"
  elif [[ $f == *"small_window"* ]]; then
    w="0.2"
  elif [[ $f == *"mlp"* ]]; then
    if [[ $f == *"window"* ]]; then
      w="0.5"
    else
      w="0.0"
    fi
  fi

  echo "Running pilot $f..."
  python vip.py --model_path $f --model_type $t --model_window_size $w --eval_mode --crash_model_path ../working_models/crash_prediction/model_1000ms_window_800ms_ahead/model --crash_model_norm_stats ../working_models/crash_prediction/model_1000ms_window_800ms_ahead/normalization_mean_std.pkl --crash_pred_window 1
done
