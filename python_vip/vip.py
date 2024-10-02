from multiprocessing import freeze_support
import sys
sys.path.append('../')
import os
import math
import random

import argparse
from datetime import datetime
from pathlib import Path

from stable_baselines3 import DDPG
from stable_baselines3 import SAC
import torch
import numpy as np
import pandas as pd
import pygame
import json
from typing import Union
import tensorflow as tf
import tensorflow.keras as keras
from transformers import InformerConfig, InformerForPrediction
from pytorch_lightning.utilities.model_summary import ModelSummary

import pickle

from deep_learning.utils.utils import (
    PandasDataset, ProcessStartField, Map, Features, Value, Sequence,
    Dataset, partial, transform_start_field, create_test_dataloader
)
from deep_learning.networks.mlp_regressor import mlp_regressor
from deep_learning.networks.rnn import rnn_regressor
from deep_learning.networks.gru import gru_regressor
from deep_learning.networks.lstm import lstm_regressor
from python_vip.Pendulum import Pendulum
from python_vip.plotting import *


import time

white = (255, 255, 255)
green = (0, 255, 0)
red = (255, 0, 0)

arrow_length = 100
arrow_head_size = 30
arrow_color = (139, 0, 0)  # Red

rl_models_type: set = {'sac', 'ddpg'}
dl_models_type: set = {'mlp', 'dl_old', 'rnn', 'gru', 'lstm'}
strategies_to_ind: dict = {
    'intervention': 0,
    'suggestion': 1,
}



def load_model(model_path: str, model_type:str, _device: str = 'cpu') -> Union[mlp_regressor, SAC, DDPG, lstm_regressor, rnn_regressor, gru_regressor, InformerForPrediction]:

    # gpu_available = torch.cuda.is_available()
    device = torch.device(_device)
    if model_type == 'mlp':
        model = mlp_regressor.load_from_checkpoint(model_path, map_location=device)
        # model = model.to(device)
    elif model_type == 'dl_old':
        model = torch.load(model_path, map_location=device)
        model.gpu = _device != 'cpu'
    elif model_type == 'rnn':
        model = rnn_regressor.load_from_checkpoint(model_path, map_location=device)
    elif model_type == 'lstm':
        model = lstm_regressor.load_from_checkpoint(model_path, map_location=device)
    elif model_type == 'gru':
        model = gru_regressor.load_from_checkpoint(model_path, map_location=device)
    elif model_type == 'sac':
        model = SAC.load(model_path)
    elif model_type == 'ddpg':
        model = DDPG.load(model_path)
    elif model_type == 'informer':
        model = InformerForPrediction.from_pretrained(model_path, device_map=device)
    else:
        raise Exception(f"type {model_type} not found. Valid options are {rl_models_type.union(dl_models_type)}")
    
    if model_type in dl_models_type:
        model = model.to(device)

    return model

def get_action_from_model(pendulum: Pendulum, model: Union[mlp_regressor, SAC, DDPG, lstm_regressor, rnn_regressor, gru_regressor], model_type: str, old_axis_x: float, window_size: int = 25, _device: str = 'cpu', return_model_input: bool=False) -> float:

    device = torch.device(_device)
    obs = None
    if model_type== 'mlp':

        if window_size > 1:
            inputs = np.array(pendulum.saved_states[-window_size:])[:, :3]
            inputs[:, 0], inputs[:, 1] = inputs[:, 1], inputs[:, 0]
            obs = inputs.reshape(-1)
            
        else:
            theta = pendulum.theta # position in degrees
            theta_dot = pendulum.theta_dot # velocity in degrees/second
            obs = np.array([theta, theta_dot, old_axis_x])
        
        axis_x = model.use(obs, device)[0]
    
    elif model_type == 'dl_old':
        theta = np.rad2deg(pendulum.theta) # position in degrees
        theta_dot = np.rad2deg(pendulum.theta_dot) # velocity in degrees/second
        obs = np.array([theta, theta_dot, old_axis_x])
        axis_x = model.use(obs)[0]

    elif model_type in {'sac', 'ddpg'}:
        x = pendulum.length/100 * np.cos(pendulum.theta)
        y = pendulum.length/100 * np.sin(pendulum.theta)
        vel = pendulum.theta_dot

        obs = [x, y, vel]
        axis_x = model.predict(obs)[0][0]
    
    elif model_type in {'rnn', 'gru', 'lstm'}:

        obs = np.array(pendulum.saved_states[-window_size:])[:, :3]
        obs[:, 0], obs[:, 1] = obs[:, 1], obs[:, 0]
        axis_x = model.use(np.array([obs]), device=device)[0][0]

    elif model_type == 'informer':
        '''
        dataframe needs the columns=['seconds', 'trialPhase', 'currentPosRollRadians', 'currentVelRollRadians', 'joystickX', 'peopleName', 'peopleTrialKey']
        '''
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ds = create_transformer_input(pendulum, model, window_size)        
        batch = next(iter(ds))

        outputs = model.generate(
            static_categorical_features=batch["static_categorical_features"].to(device)
            if model.config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"].to(device)
            if model.config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"].type(torch.FloatTensor).to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].type(torch.FloatTensor).to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
        )

        forecasts = outputs.sequences.cpu().numpy()
        forecasts = np.mean(forecasts, axis=1)
        # forecasts = np.median(forecasts, axis=1)
        axis_x = forecasts[0][0]

        # axis_x = np.random.rand()
    if return_model_input:
        return np.clip(axis_x, -1, 1), obs
    else:
        return np.clip(axis_x, -1, 1)


def create_transformer_input(pendulum: Pendulum, model: InformerForPrediction, window_size: int):
    
    effective_window_size = window_size

    raw_dataframe = pd.DataFrame(
        {
            'seconds': pendulum.times[-effective_window_size:], 
            'trialPhase': [3]*effective_window_size, 
            'currentPosRollRadians': pendulum.positions[-effective_window_size:], 
            'currentVelRollRadians': pendulum.velocities[-effective_window_size:],
            'joystickX': pendulum.joystick_actions[-effective_window_size:],
            'peopleName': ['pyvip-trial-transformer']* effective_window_size,
            'peopleTrialKey': ['pyvip-trial-0'] * effective_window_size,
            'peopleTrialKey_window_num': ['pyvip-trial-1'] * effective_window_size
        })

    freq='S'

    test_ds = PandasDataset.from_long_dataframe(raw_dataframe, target="joystickX", item_id="peopleTrialKey_window_num", timestamp="seconds", feat_dynamic_real=["currentPosRollRadians", "currentVelRollRadians"], freq="ms", unchecked=True)

    process_start = ProcessStartField()
    list_test_ds = list(Map(process_start, test_ds))

    features  = Features(
        {    
            "start": Value("timestamp[s]"),
            "target": Sequence(Value("float32")),
            "feat_static_cat": Sequence(Value("uint64")),
            # "feat_static_real":  Sequence(Value("float64")),
            "feat_dynamic_real": Sequence(Sequence(Value("float32"))),
            # "past_feat_dynamic_real": Sequence(Sequence(Value("float32"))),
            "item_id": Value("string"),
        }
    )
    
    test_dataset = Dataset.from_list(list_test_ds, features=features)

    test_dataset.set_transform(partial(transform_start_field, freq=freq))

    # lags_sequence = get_lags_for_frequency(freq, lag_ub=30)
    # time_features = time_features_from_frequency_str(freq)

    test_dataloader = create_test_dataloader(
        config=model.config,
        freq=freq,
        data=test_dataset,
        batch_size=64,
    )



    return test_dataloader



def get_crash_probability(model: keras.Sequential, pendulum: Pendulum, crash_norm_stats, window_size: int = 50) -> float:
    inputs = np.array(pendulum.saved_states[-window_size:])
    inputs[:, 0] = (np.degrees(inputs[:, 0]) - crash_norm_stats[0]['mean']) / crash_norm_stats[0]['std']
    inputs[:, 1] = (np.degrees(inputs[:, 1]) - crash_norm_stats[1]['mean']) / crash_norm_stats[1]['std']
    inputs[:, 2] = (np.degrees(inputs[:, 2]) - crash_norm_stats[2]['mean']) / crash_norm_stats[2]['std']

    inputs = tf.convert_to_tensor([inputs], dtype=tf.float32)
    crash_prob = model.predict(inputs, verbose=0)[0][0]

    return crash_prob


def print_model_summary(model: Union[mlp_regressor, rnn_regressor, lstm_regressor, gru_regressor, keras.Sequential], window_size: int, model_name: str) -> None:
    
    print(f"Loaded model: {model_name}")

    if type(model) == keras.Sequential:
        print(model.summary())
    else:
        # input_size = (1, 3*(window_size+1)) if type(model) == mlp_regressor else (1, window_size+1, 3)
        print(ModelSummary(model))
    
    return
        

def main():
    np.random.seed(42)

    parser = argparse.ArgumentParser(
                        prog='Python Virtual Inverted Pendulum (PyVIP)',
                        description='A VIP simulation where a human or a provided model is asked to control the Inverted pendulum and prevent it from falling',
                        epilog='IDK what this does')

    parser.add_argument('--protocol', default='./protocols.csv', required=False)
    parser.add_argument('--experiment_config', default=None, required=False, type=str)
    parser.add_argument('--model_path', default=None, required=False, type=str)
    parser.add_argument('--model_type', default='sac', required=False, type=str, choices=['sac', 'ddpg', 'mlp', 'dl_old', 'rnn', 'gru', 'lstm', 'informer'])
    parser.add_argument('--model_window_size', required=False, default=0, type=float)
    parser.add_argument('--model_intermittent', default=0, type=int)
    parser.add_argument('--noise_model_action', required=False, type=float, default=0)
    parser.add_argument('--eval_mode', action='store_true')
    parser.add_argument('--crash_model_path', required=False, type=str)
    parser.add_argument('--crash_model_norm_stats', type=str, required=False)
    parser.add_argument('--crash_pred_window', required=False, type=float, default=1)
    parser.add_argument('--device', required=False, default='cpu', type=str)
    parser.add_argument('--show_crash_bounds', required=False, default=False, action='store_true')
    parser.add_argument('--run_hitl', required=False, default=False, action='store_true')
    parser.add_argument('--experiment_name', required=False, default="experiment_1", type=str)
    parser.add_argument('--study_name', required=False, default=None, type=str)
    parser.add_argument('--ppt_id', required=False, default="", type=str)
    parser.add_argument('--catch_trial', required=False, action="store_true", default=False)
    parser.add_argument('--monitor', required=False, default=0, type= int)
    parser.add_argument('--show_metadata', required=False, default=False, action='store_true')
    # parser.add_argument('--plot_post_performance', required=False, default=False, action='store_true')
    parser.add_argument('--use_joystick', required=False, default=False, action='store_true')
    # parser.add_argument('--study_name', required=False, default="", type=str)
    args = parser.parse_args()


    device = args.device
    monitor = args.monitor
    use_joystick = args.use_joystick
    model_path: str = args.model_path
    model_intermittent: int = args.model_intermittent / 100
    model: Union[mlp_regressor, SAC, DDPG] = None
    model_name: str = ''

    exp_config_path: str = args.experiment_config
    exp_config: dict = None
    run_exp = False
    run_hitl = False
    pilot: str = ''
    pilot_model: Union[mlp_regressor, SAC, DDPG, lstm_regressor, rnn_regressor, gru_regressor] = None
    assistant: str = ''
    assistant_model: Union[mlp_regressor, SAC, DDPG, lstm_regressor, rnn_regressor, gru_regressor] = None 
    strategy: int = 0

    crash_model: keras.Sequential = None
    crash_norm_stats: dict = None

    pilot_model_window_size = 0
    assistant_model_window_size = 0
    crash_pred_window_size = 0

    if exp_config_path:
        run_exp = True
        with open(exp_config_path, 'r') as f:
            exp_config = json.load(f)
            config = exp_config['pyvip_config']
            
            strategy = strategies_to_ind[config['strategy']]

            pilot = config['pilot']['type']
            pilot_name = config['pilot']['name']
            if pilot != 'human':
                pilot_model = load_model(config['pilot']['path'], config['pilot']['type'], device)
                if 'window_size' in config['pilot']:
                    pilot_model_window_size = int(config['pilot']['window_size'] * 50) + 1

                if pilot in dl_models_type:
                    print_model_summary(pilot_model, pilot_model_window_size, pilot_name)


            assistant = config['assistant']['type']
            assistant_name = config['assistant']['name']
            if assistant != 'human':
                assistant_model = load_model(config['assistant']['path'], config['assistant']['type'], device)
                if 'window_size' in config['assistant']:
                    assistant_model_window_size = int(config['assistant']['window_size'] * 50) + 1
            
                if assistant in dl_models_type:
                    print_model_summary(assistant_model, assistant_model_window_size, assistant_name)
                

            if "crash_predictor" in config:
                crash_model = keras.models.load_model(config['crash_predictor']['model_path'])
                with open(config['crash_predictor']['norms_path'], 'rb') as f:
                    crash_norm_stats = pickle.load(f)
                crash_pred_window_size = int(config['crash_predictor']['window_size_in_seconds'] * 50)

                print_model_summary(crash_model, crash_pred_window_size, "crash_predictor")


        model_name = f"strategy_{config['strategy']}_pilot_{pilot_name}_assistant_{assistant_name}"

        model_window_size: int = 0
    else :
        
        if args.run_hitl:
            run_hitl = True
        if model_path:
            model_type = args.model_type
            model = load_model(model_path, model_type, device)
            model_name = model_path.split("/")[-1] 

        if args.model_intermittent: 
            model_name += f"_intermittent_{args.model_intermittent}" 

        if args.crash_model_path and args.crash_model_norm_stats:
            crash_model = keras.models.load_model(args.crash_model_path)
            with open(args.crash_model_norm_stats, 'rb') as f:
                crash_norm_stats = pickle.load(f)
            crash_pred_window_size = int(args.crash_pred_window * 50)

            model_name += f"_crash_predictor_{model_path.split('/')[-2] if model_path else ''}"

        config = None
        crash_pred_window_size: int = int(args.crash_pred_window * 50)
        
        model_window_size: int = int(args.model_window_size * 50) + 1
        pilot_model_window_size = 0
        assistant_model_window_size = 0

    
    experiment_name = args.experiment_name
    dir_prefix = f"{args.study_name}/{args.ppt_id}/{experiment_name}" if args.study_name else \
        f"hitl_human_{experiment_name}_model_{model_name}" if run_hitl else  \
        model_name if model_path or run_exp else f"{experiment_name}" 

    output_dir = f'../output/{dir_prefix}/{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    protocols_file: str = args.protocol
    # print(os. getcwd())
    protocols = pd.read_csv(protocols_file)
    protocols = protocols.fillna(method='ffill')


    eval_mode = args.eval_mode

    show_crash_bounds = args.show_crash_bounds
    show_metadata = args.show_metadata
    # plot_info_realtime = args.plot_info_realtime
    pygame.font.init()
    font = pygame.font.SysFont('didot.ttc', 24)

    is_catch_trial = []
    if args.catch_trial:
        is_catch_trial = [True] + [False] * (len(protocols) - 1)
        random.shuffle(is_catch_trial)
    else:
        is_catch_trial = [False] * len(protocols)

    
    INFO_TEXT = ""


    # Initialize pygame

    for index, row in protocols.iterrows():
        pygame.font.init()
        font = pygame.font.SysFont('didot.ttc', 24)

        START_TEXT = f"Press space to start run_number {index}"
        CRASH_TEXT = f"You have crashed. {START_TEXT.replace('start', 'continue')}."
        START_RECT = font.render(START_TEXT, True, green, white)
        CRASH_RECT = font.render(CRASH_TEXT, True, red, white)
        output_dir = f'../output/{dir_prefix}/{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}/'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        pygame.init()
        pygame.event.clear()
        # Set up the display
        FPS: int = row['fps']
        MAX_TRIAL_TIME: int = row['trial_time']
        POS_LIMIT: float = np.radians(row['position_limit'])
        DOT_RADIUS = 10
        n_dots = int(row['ndots'])
        coherence = int(row['coherence'])
        size = (600, 600)
        screen = pygame.display.set_mode(size, display=monitor)
        pygame.display.set_caption("Inverted Pendulum")
        base_width = 100
        base_height = 20

        # bv = 10
        gr: float = row['gravity']
        ksp: float = row['ksp']
        length: float = row['length']
        bv: float = row['bv']
        mass: float = row['mass']
        ipoff: float = row['ipoff']
        MAX_WINDOW_SIZE : int =  int(max(crash_pred_window_size, model_window_size, pilot_model_window_size, assistant_model_window_size)) 
        times: list[float] = [0] * MAX_WINDOW_SIZE
        positions: list[float] = [0] * MAX_WINDOW_SIZE
        velocities: list[float] = [0] * MAX_WINDOW_SIZE
        joystick_actions: list[float] = [0] * MAX_WINDOW_SIZE
        destab_actions: list[float] = [0] * MAX_WINDOW_SIZE
        saved_states: list[list[float]] = [[0, 0, 0, 0]] * MAX_WINDOW_SIZE # [[vel, pos, joystick, destab]]


        is_crash_condition_triggered: list[int] = [0]
        pilot_actions: list[float] = [0]
        assistant_actions: list[float] = [0]
        who_made_the_action: list[int] = [0] # 1 by pilot, 3 by assistant
        crash_probabilities: list[float] = [0] # 0 no crash, 1 crash
        # Define the Pendulum class

        # Set up the joystick
        joystick = None
        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()

        # Initialize the pendulum with random starting position
        pendulum = Pendulum(length, mass, times, positions, velocities, joystick_actions, destab_actions, saved_states, ipoff=ipoff)
        dots, P, R, nchg, last_pos = False, None, None, 0, pendulum.theta
        if n_dots > 0:
            dots = True
            # create dots in cartesian coordinates
            p = np.random.rand(n_dots, 1) * 2 * np.pi
            r = np.sqrt(np.random.rand(n_dots, 1))

            P = np.zeros((n_dots, 2))
            P[:, 0] = np.multiply(r, np.sin(p)).reshape(-1)
            P[:, 1] = np.multiply(r, np.cos(p)).reshape(-1)

            nchg = int(n_dots - np.floor(n_dots * coherence / 100))


        
        SAFE_ZONE: float = POS_LIMIT * config['SAFE_ZONE'] if config else 1/5 # 0-12 degrees from the center is safe
        DANGER_ZONE: float = POS_LIMIT * config['DANGER_ZONE'] if config else 1/3 # 40-60 degrees from the center is considered dangerous 
        NOISE_DEVIATION_SUGGESTION: float = config['NOISE_DEVIATION_SUGGESTION'] if config else args.noise_model_action
        ASSISTANT_CONFIDENCE: float = config['ASSISTANT_CONFIDENCE'] if config else 1
        HUMAN_REACTION_TIME: float = config['HUMAN_REACTION_TIME'] if config else 0.4
        HUMAN_REACTION_TIME_NOISE: float = config['HUMAN_REACTION_TIME_NOISE'] if config else 0.05
        CRASH_PROB_THRESHOLD: float = config['CRASH_PROB_THRESHOLD'] if config else 0.5
        DISAGREEMENT_EPISODES = [] # stores [input, model_action, human_action]

        delayed_suggestions: list[float] = []
        human_reaction_delay: int = 0

        start_simulation = eval_mode
        # Set up the clock
        clock = pygame.time.Clock()
        total_time_elapsed: int = 0
        time_elapsed: int = 0

        catch_suggestions = []
        is_catch = is_catch_trial[index]
        # Game loop
        axis_x: float = 0 # previous joystick value
        check_destab_action = lambda values : np.all(values != 0) and np.all(np.sign(values) == np.sign(values[0]))

        human_action: float = 0 # previous human action
        model_action: float = 0 # previous model action

        crash_prob: float = 0
        check = False
        while True:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            if run_exp:
                '''
                get joystick from pilot whether human or not
                if about to crash: depending on the strategy
                if strategy is to intervene: assistant model will 
                '''
                # check if about to crash
                if pilot_model:
                    pilot_action = get_action_from_model(pendulum, pilot_model, pilot, axis_x, pilot_model_window_size, device)
                if assistant_model:
                    assistant_action = get_action_from_model(pendulum, assistant_model, assistant, axis_x, assistant_model_window_size, device)

                # Checking to see whether the pilot is in a dangerous position 
                check = False
                if crash_model:
                    #check if will crash
                    crash_prob = get_crash_probability(crash_model, pendulum, crash_norm_stats, crash_pred_window_size)
                    check = crash_prob > CRASH_PROB_THRESHOLD 
                    
                    crash_probabilities.append(crash_prob)
                else:
                    check = abs(pendulum.theta) > SAFE_ZONE
                    crash_probabilities.append(crash_prob)

                # if in danger of crashing or outside safe zone 
                check = (check and abs(pendulum.theta) > SAFE_ZONE) or abs(pendulum.theta) > DANGER_ZONE
                if check and strategy == 1:
                    if pilot != 'human':
                        if ((time_elapsed - human_reaction_delay) /1000) >= (HUMAN_REACTION_TIME + np.random.uniform(-HUMAN_REACTION_TIME_NOISE, HUMAN_REACTION_TIME_NOISE)) :
                            is_crash_condition_triggered.append(0)

                            if np.random.rand() > ASSISTANT_CONFIDENCE:
                                # use pilot's value
                                axis_x = pilot_action + np.random.uniform(-NOISE_DEVIATION_SUGGESTION, NOISE_DEVIATION_SUGGESTION)
                                if np.random.rand() < model_intermittent :
                                    axis_x = 0
                                who_made_the_action.append(1)
                                
                            else:
                                # use the delayed suggested value within a deviation 
                                if delayed_suggestions:
                                    axis_x = delayed_suggestions.pop(0)
                                else:
                                    axis_x = assistant_actions
                                    human_reaction_delay = time_elapsed
                                axis_x = delayed_suggestions.pop(0) if delayed_suggestions else assistant_action
                                # adding some noise since a person is not perfect
                                axis_x = axis_x + np.random.uniform(-NOISE_DEVIATION_SUGGESTION, NOISE_DEVIATION_SUGGESTION)
                                if np.random.rand() < model_intermittent :
                                    axis_x = 0
                                
                                who_made_the_action.append(3)

                        else:
                            is_crash_condition_triggered.append(1)

                            # use pilot's action
                            # if the human is still reacting to the delay then
                            # save the suggested action by the assistant and apply it after a delay that simulates the human reaction time. 
                            delayed_suggestions.append(assistant_action)
                            axis_x = pilot_action + np.random.uniform(-NOISE_DEVIATION_SUGGESTION, NOISE_DEVIATION_SUGGESTION)
                            if np.random.rand() < model_intermittent :
                                axis_x = 0
                            who_made_the_action.append(1)
                    else:
                        # also need to add how to suggest the action to the human.
                        is_crash_condition_triggered.append(1)
                        

                        if joystick is not None:
                            axis_x = joystick.get_axis(0)
                            if abs(axis_x) < 0.1:
                                axis_x = 0
                        else:
                            keys = pygame.key.get_pressed()
                            if keys[pygame.K_LEFT]:
                                axis_x = -1
                            elif keys[pygame.K_RIGHT]:
                                axis_x = 1
                            else:
                                axis_x = 0

                        pilot_action = axis_x # only when an actual human is performing 

                        who_made_the_action.append(4) # 4 represents when human may or may not accept the suggestion since we have no way to actually know during the run. Can do a postmortem analysis on both actions direction's but still doesn't verify whether the assistant's suggestion was accepted. 
                        

                elif check and strategy == 0: 
                    is_crash_condition_triggered.append(1)
                    INFO_TEXT = 'You were about to crash, assistant is intervening'
                    if assistant == 'human':
                        if joystick is not None:
                            axis_x = joystick.get_axis(0)
                            if abs(axis_x) < 0.1:
                                axis_x = 0
                        else:
                            keys = pygame.key.get_pressed()
                            if keys[pygame.K_LEFT]:
                                axis_x = -1
                            elif keys[pygame.K_RIGHT]:
                                axis_x = 1
                            else:
                                axis_x = 0
                    
                        assistant_action = axis_x
                    else:
                        axis_x = assistant_action
                        if np.random.rand() < model_intermittent :
                            axis_x = 0
                        
                    who_made_the_action.append(3)

                else:
                    # pilot will continually give input
                    # human_reaction_delay = time_elapsed
                    # if delayed_suggestions:
                    #     delayed_suggestions.clear()
                    is_crash_condition_triggered.append(0)
                    if pilot == 'human':
                        if joystick is not None:
                            axis_x = joystick.get_axis(0)
                            if abs(axis_x) < 0.1:
                                axis_x = 0
                        else:
                            keys = pygame.key.get_pressed()
                            if keys[pygame.K_LEFT]:
                                axis_x = -1
                            elif keys[pygame.K_RIGHT]:
                                axis_x = 1
                            else:
                                axis_x = 0

                        pilot_action = axis_x # only when an actual human is performing 
                    else:
                        axis_x = pilot_action + np.random.uniform(-NOISE_DEVIATION_SUGGESTION, NOISE_DEVIATION_SUGGESTION)
                        if np.random.rand() < model_intermittent :
                            axis_x = 0
                    
                    who_made_the_action.append(1)
                

                pilot_actions.append(pilot_action)
                assistant_actions.append(assistant_action)

            elif run_hitl:
                if crash_model:
                    crash_prob = get_crash_probability(crash_model, pendulum, crash_norm_stats,crash_pred_window_size)

                who_made_the_action.append(1)
                crash_probabilities.append(crash_prob)
                # Read input from the joystick or keyboard
                
                model_action, model_input = get_action_from_model(pendulum, model, model_type, axis_x, model_window_size, device, return_model_input=True) 
                model_action += np.random.uniform(-NOISE_DEVIATION_SUGGESTION, NOISE_DEVIATION_SUGGESTION)

                if np.random.rand() < model_intermittent :
                    model_action = 0

                if joystick is not None:
                    human_action = joystick.get_axis(0)
                    if abs(human_action) < 0.1:
                        human_action = 0
                else:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_LEFT]:
                        human_action = -1
                    elif keys[pygame.K_RIGHT]:
                        human_action = 1
                    else:
                        human_action = 0
                
                if human_action != 0:
                    if np.sign(human_action) != np.sign(model_action):
                        axis_x = model_action * -1
                        is_crash_condition_triggered.append(1)
                        DISAGREEMENT_EPISODES.append([model_input, model_action, human_action])
                    else:
                        is_crash_condition_triggered.append(0)
                        axis_x = model_action
                else:
                    is_crash_condition_triggered.append(0)
                    axis_x = model_action

                pilot_actions.append(model_action)
                assistant_actions.append(human_action)
            else:
                is_crash_condition_triggered.append(0)
                # running the program with a human or model
                if crash_model:
                    crash_prob = get_crash_probability(crash_model, pendulum, crash_norm_stats, crash_pred_window_size)

                who_made_the_action.append(1)
                crash_probabilities.append(crash_prob)
                # Read input from the joystick or keyboard
                if use_joystick and joystick is not None:
                    axis_x = joystick.get_axis(0)
                    if abs(axis_x) < 0.1:
                        axis_x = 0
                elif model_path:

                    axis_x = get_action_from_model(pendulum, model, model_type, axis_x, model_window_size, device) + np.random.uniform(-NOISE_DEVIATION_SUGGESTION, NOISE_DEVIATION_SUGGESTION)

                    if np.random.rand() < model_intermittent :
                        axis_x = 0

                else:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_LEFT]:
                        axis_x = -1
                    elif keys[pygame.K_RIGHT]:
                        axis_x = 1
                    else:
                        axis_x = 0
                
                pilot_actions.append(axis_x)
                assistant_actions.append(0.0)
            axis_x = np.clip(axis_x, -1, 1)

            # Update the pendulum
            if axis_x == 0:
                # If there is no input, let the pendulum fall towards the crash boundary under the influence of gravity
                # pendulum.theta_dot = 0
                pendulum.theta += math.copysign(1, math.sin(pendulum.theta)) * 0.01
                # pass
            else:
                # Otherwise, apply the input torque to the pendulum
                pendulum.theta_dot += axis_x * 0.1
            
            joystick_actions.append(axis_x)
            # check destabilizing action

            is_destab_action = check_destab_action([axis_x, pendulum.theta, pendulum.theta_dot])
            destab_actions.append(is_destab_action)

            saved_states.append([pendulum.theta_dot, pendulum.theta, axis_x, is_destab_action])

            pendulum_tip_x, pendulum_tip_y = pendulum.update(base_x=screen.get_width() / 2, base_y=screen.get_height() - 50, dt=0.005, time_elapsed=time_elapsed, ksp=ksp, bv=bv, gr=gr, base_height=base_height)

            # Clear the screen
            screen.fill((255, 255, 255))
            base_center_x = screen.get_width() / 2
            base_center_y = screen.get_height() - 50 - base_height / 2
            if not dots:
            # Draw the pendulum
                pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(base_center_x - base_width / 2, base_center_y, base_width, base_height))
                pygame.draw.line(screen, (0, 0, 0), (base_center_x, base_center_y), (pendulum_tip_x, pendulum_tip_y), 5)
                pygame.draw.circle(screen, (0, 0, 0), (int(pendulum_tip_x), int(pendulum_tip_y)), 10)
                
            else:

                rotinc = -(pendulum.theta - last_pos)
                R = np.array([[np.cos(rotinc), -np.sin(rotinc)], [np.sin(rotinc), np.cos(rotinc)]]) 

                P = np.dot(P, R)
                
                if nchg > 0:
                    p = np.random.rand(nchg, 1) * 2 * np.pi
                    r = np.sqrt(np.random.rand(nchg, 1))
                    P[:nchg, 0] = np.multiply(r, np.sin(p)).reshape(-1)
                    P[:nchg, 1] = np.multiply(r, np.cos(p)).reshape(-1)

                for dot in P:
                    # print(dot)
                    pygame.draw.circle(screen, (0, 0, 0), (dot[0]*(2*pendulum.length+20) + base_center_x, dot[1]*(2*pendulum.length+20) + base_center_y), 
                    DOT_RADIUS)
                
                last_pos = pendulum.theta
                np.random.shuffle(P)
            
            catch_sug = is_catch and np.random.rand() >= 0.5 
            catch_suggestions.append(catch_sug)
            if catch_sug or check and strategy == 1 :
                assistant_direction_suggestion = np.sign(assistant_action)
                if catch_sug:
                    assistant_direction_suggestion = 1 if np.random.rand() >= 0.5 else -1

                if assistant_direction_suggestion == 1:
                    arrow_x = screen.get_width() - arrow_head_size
                    pygame.draw.polygon(screen, arrow_color, [(arrow_x, arrow_head_size + arrow_length), (arrow_x - arrow_head_size, arrow_head_size + arrow_length - arrow_head_size), (arrow_x - arrow_head_size, arrow_head_size + arrow_length + arrow_head_size)])
                elif assistant_direction_suggestion == -1:
                    arrow_x = 20
                    pygame.draw.polygon(screen, arrow_color, [(arrow_x, arrow_head_size + arrow_length), (arrow_x + arrow_head_size, arrow_head_size + arrow_length - arrow_head_size), (arrow_x + arrow_head_size, arrow_head_size + arrow_length + arrow_head_size)])

            if show_crash_bounds:
                base_x=screen.get_width() / 2 
                base_y=screen.get_height() - 50

                crash_bound_x = base_x + pendulum.length/100*200 * math.sin(POS_LIMIT)
                crash_bound_y = base_y - base_height / 2 - (pendulum.length/100*200) * math.cos(POS_LIMIT)

                pygame.draw.line(screen, (255, 0, 0), (base_center_x, base_center_y), (crash_bound_x, crash_bound_y), 1)
                
                crash_bound_x = base_x + pendulum.length/100*200 * math.sin(-POS_LIMIT)
                crash_bound_y = base_y - base_height / 2 - (pendulum.length/100*200) * math.cos(-POS_LIMIT)

                pygame.draw.line(screen, (255, 0, 0), (base_center_x, base_center_y), (crash_bound_x, crash_bound_y), 1)
            

            pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(0, 0, 600, 20))
            text = font.render(f"Time Left: {MAX_TRIAL_TIME - ((time_elapsed + total_time_elapsed)/1000):0.3f} seconds", True, (0, 0, 0))
            screen.blit(text, (20, 40))
            if show_metadata:
                
                text = font.render(f"Position: {np.degrees(pendulum.theta):0.3f} degrees", True, (0, 0, 0))
                screen.blit(text, (20, 60))
                text = font.render(f"Velocity: {np.degrees(pendulum.theta_dot):0.3f} degrees/s", True, (0, 0, 0))
                screen.blit(text, (20, 80))
                text = font.render(f"Joystick deflection: {axis_x:0.3f}", True, (0, 0, 0))
                screen.blit(text, (20, 100))
                text = font.render(f"Crash probability: {crash_prob:0.3f}", True, (0, 0, 0))
                screen.blit(text, (20, 120))
                text = font.render(f"User action: {human_action:0.3f}", True, (0, 0, 0))
                screen.blit(text, (20, 140))
                text = font.render(f"AI action: {model_action:0.3f}", True, (0, 0, 0))
                screen.blit(text, (20, 160))


            # Check if the pendulum has fallen
            if abs(pendulum.theta) >= POS_LIMIT:
                print("Crash!")
                pendulum = Pendulum(length, mass, pendulum.times, pendulum.positions, pendulum.velocities, pendulum.joystick_actions, pendulum.destab_actions, pendulum.saved_states, ipoff=ipoff, time_elapsed=time_elapsed)
                start_simulation = False
                who_made_the_action.append(0)
                pilot_actions.append(0.0)
                assistant_actions.append(0.0)
                crash_probabilities.append(1)
                is_crash_condition_triggered.append(0)
                saved_states.extend([[0,0,0,0]] * MAX_WINDOW_SIZE)
                axis_x = 0
                start_simulation = eval_mode

                while not start_simulation:
                    pygame.draw.rect(CRASH_RECT, white, CRASH_RECT.get_rect(), 1)
                    screen.blit(CRASH_RECT, (20, 20))
                    pygame.display.update()

                    event = pygame.event.wait()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        start_simulation = True
                pygame.display.flip()
                clock.tick(FPS)
                continue

            # Flip the display
            pygame.display.flip()

            while not start_simulation:
                pygame.draw.rect(START_RECT, white, START_RECT.get_rect(), 1)
                screen.blit(START_RECT, (20, 20))
                pygame.display.update()
                event = pygame.event.wait()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    start_simulation = True
                pygame.display.flip()
                clock.tick(FPS)
                continue

            # Wait for a short amount of time to control the frame rate
            time_elapsed += clock.tick(FPS)
            if ((time_elapsed + total_time_elapsed) /1000) >= MAX_TRIAL_TIME:
                pygame.quit()
                break

        
        # save plots and data from experiment

        # saving plot for position, velocity and joystick deflection with time

        if run_hitl:
            print(f"human and ai disagreed {len(DISAGREEMENT_EPISODES)} times")
            episodes = np.array(
                [np.append(x[0] if model_type in ['sac', 'ddpg'] else x[0].reshape(-1), x[1:]) for x in DISAGREEMENT_EPISODES]
                )
            np.savetxt(f"{output_dir}disagreement_episodes.txt", episodes, delimiter=',')


        if is_catch and len(catch_suggestions):
            meta_info = {
                "catch_trial": [index],
                "catch_suggestions": catch_suggestions
            }
            with open(f'{output_dir}meta_info_catch.json', 'w') as f:
                json.dump(meta_info, f)

        times = np.array(pendulum.times[MAX_WINDOW_SIZE:]).reshape((-1,1)) / 1000
        positions = np.degrees(np.array(pendulum.positions[MAX_WINDOW_SIZE:]).reshape((-1,1)))
        velocities = np.degrees(np.array(pendulum.velocities[MAX_WINDOW_SIZE:]).reshape((-1,1)))
        joystick_actions = np.array(pendulum.joystick_actions[MAX_WINDOW_SIZE:]).reshape((-1,1))
        who_made_the_action = np.array(who_made_the_action).reshape((-1,1))
        pilot_actions = np.array(pilot_actions).reshape((-1,1))
        assistant_actions = np.array(assistant_actions).reshape((-1,1))
        destab_actions = np.array(pendulum.destab_actions[MAX_WINDOW_SIZE:]).reshape((-1, 1))
        crash_probabilities = np.array(crash_probabilities).reshape((-1,1))
        is_crash_condition_triggered = np.array(is_crash_condition_triggered).reshape(-1, 1)

        output_file_prefix = ''

        if dots:
            output_file_prefix += f'dots_{n_dots}_coherence_{coherence}'
        else:
            output_file_prefix += 'pendulum_only'

        # saving trial plot wit pos, vel and actions
        plot_trial_normal(times, positions, velocities, joystick_actions, f'{output_dir}{output_file_prefix}_trial_plot.png')

        # saving plot for who made the action during the trial
        plt.clf()
        print(len(times), len(who_made_the_action))

        plot_trial_who_made_action(times, who_made_the_action, f"{output_dir}{output_file_prefix}_actions_log_plot.png")

        # saving plot what action was made by the pilot and assistant during the trial
        plt.clf()
        plot_trial_normal_entities(times, positions, pilot_actions, assistant_actions, f'{output_dir}{output_file_prefix}_trial_plot_actions.png')
        
        # saving plot for crash probs with respect to pos and vel
        plt.clf()
        plot_trial_crash_prob(times, positions, velocities, crash_probabilities, f'{output_dir}{output_file_prefix}_trial_plot_crash_probs.png')

        # plt.clf()
        # plot_trial_pos_vel(times, positions, velocities, destab_actions, f'{output_dir}{output_file_prefix}_trial_plot_pos_vel.png')

        # need to save experiment details e.g. constants, noises etc. 
        
        # saving raw data 
        data = np.hstack((times, positions, velocities, joystick_actions, who_made_the_action, pilot_actions, assistant_actions, destab_actions, crash_probabilities, is_crash_condition_triggered))
        dataframe = pd.DataFrame(data, columns=['time', 'angular position', 'angular velocity', 'joystick deflection', 'action_made_by', 'pilot_actions', 'assistant_actions', 'destabilizing_actions', 'crash_probabilities', 'is_crash_condition_triggered'])
        dataframe.to_csv(f'{output_dir}{output_file_prefix}_trial_data.csv', index=False)

    pygame.display.quit()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    # freeze_support()
    main()