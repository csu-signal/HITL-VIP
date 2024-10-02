import sys
sys.path.append("../")
import gymnasium as gym
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import Dataset, random_split

from deep_learning.utils.utils import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

import os
import torch
import pandas as pd

from python_vip.vip import load_model, print_model_summary



from torch.utils.data import DataLoader

from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from imitation.data.types import Transitions
from stable_baselines3.common.env_util import make_vec_env
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm

from reinforcement.envs.pendulumV21 import PendulumEnv

from reinforcement.envs.pendulum_penalizeDeflection import PendulumEnv as penalizeDeflection

import glob
import numpy as np





def get_model_id(mappings, ppt, session_num):
    if ppt in set(mappings["participant"]):
        # Find the row index where X is present in column A
        row_index = mappings.index[mappings["participant"] == ppt].tolist()[0]
        # Get the corresponding value from column B
        corresponding_value = mappings.loc[row_index, "model"] if session_num == 1 else mappings.loc[row_index, "model_second_session"]
        return corresponding_value
    else:
        print(f"No occurrence of '{ppt}' found in participants columns.")
        return None


def num_helpful_suggestions(df: pd.DataFrame, alone=True):

    count_followed = 0
    count_recovered = 0
    count_crashed = 0
    time_to_recover = []
    time_to_crash = []

    suggestion_started = False
    start_time = 0
    start_pos = 0
    curr_time = 0
    suggestion_direction = 0
    start_index = 0
    
    for i in range(len(df)):

        if not suggestion_started:
            if not alone:
                suggestion_started = df.iloc[i]["action_made_by"] == 4 and abs(df.iloc[i]["angular position"]) > np.degrees(0.2)
            else:
                suggestion_started = abs(df.iloc[i]["angular position"]) > np.degrees(0.2)
            if suggestion_started:
                start_time = df.iloc[i]["time"]
                curr_time = start_time
                start_pos = df.iloc[i]["angular position"]
                start_index = i
                suggestion_direction = np.sign(df.iloc[i]["assistant_actions"])
        
        else:

            # check position: if returned to safe region, update counts elif still not safe, continue, if crashed update counts and continue
            if not alone:
                if curr_time - df.iloc[i]["time"] <= 0.45:
                    if suggestion_direction == np.sign(df.iloc[i]["pilot_actions"]):
                        count_followed += 1
                else:
                    start_index += 1
                    curr_time = df.iloc[start_index]["time"]
                    suggestion_direction = np.sign(df.iloc[start_index]["assistant_actions"])
            
            
            
            pos = df.iloc[i]["angular position"]
            recovered = abs(pos) < np.degrees(0.2) and abs(pos) < abs(start_pos)
            crashed = abs(pos) >= 60
            if recovered:
                suggestion_started = False
                time_to_recover.append(df.iloc[i]["time"] - start_time)
                count_recovered += 1
            elif crashed :
                suggestion_started = False
                time_to_crash.append(df.iloc[i]["time"] - start_time)
                count_crashed += 1

            

        

    if len(time_to_crash) == 0:
        time_to_crash.append(0)
    if len(time_to_recover) == 0:
        time_to_recover.append(0)

    return (
        count_recovered,
        np.average(time_to_recover),
        count_crashed,
        np.average(time_to_crash),
        count_followed
    )




def learning_effect(data):
    dist_score = (60 - abs(data["mean_dist_dob"].values[0])) / 60
    defl_score = (1 - abs(data["defl_mag_mean"].values[0])) / 1
    crash_freq_score = 1 - data["crash_freqs"].values[0]
    destab_score = 1 -  (data["perc_destab_actions"].values[0] / 100)
    anticip_score = data["perc_anticip_actions"].values[0] / 100
    recovery_count = data.count_recovered.values[0]
    crashed_count = data.count_crashed.values[0]
    recovery_score = (recovery_count) - (crashed_count if crashed_count != 0 else 1)

    score = sum([
       dist_score, defl_score, crash_freq_score, destab_score, anticip_score, recovery_score 
    ])

    return score

def get_data_stats(data):
    check_anticipatory_deflection = lambda x: 1 if x[0]!=0 and np.sign(x[1])!=np.sign(x[2])  else 0
    data['anticipatory_deflections'] = data.apply(lambda x: check_anticipatory_deflection([x[1], x[2], x[3]]), axis=1)
    assign_label = lambda x,y: 0 if (x == y) or (not x and not y) else 1 if x == 1 else 2
    data['deflection_type_label'] = data.apply(lambda x: assign_label(x[7], x[10]), axis=1)


    times = np.array(data['time'])
    num_actions = np.size(data["destabilizing_actions"])
    num_destabilizing_actions = np.sum(data["destabilizing_actions"])
    destabilizing_prop = num_destabilizing_actions/num_actions
    num_crash_cond_triggers = np.sum(data["is_crash_condition_triggered"])
    crash_cond_triggered_prop = num_crash_cond_triggers / num_actions
    n_crashes = np.sum(np.array(data["action_made_by"]) == 0.0)-1
    last_timestamp = times[-1]
    crash_freq = n_crashes/last_timestamp
    avg_crash_probability = np.mean(np.array(data["crash_probabilities"]))
    avg_dob_dist = np.mean(np.abs(np.array(data["angular position"])))
    sd_angular_pos = np.std(np.array(data["angular position"]))
    avg_angvel_mag = np.mean(np.abs(np.array(data["angular velocity"])))
    sd_angvel = np.std(np.array(data["angular velocity"]))
    angvel_rms = np.sqrt(np.mean(np.array(data["angular velocity"])**2))
    avg_defl_mag = np.mean(np.abs(np.array(data["joystick deflection"])))
    # frequency_data = len(times)/last_timestamp
    
    prop_anticipatory_deflections = np.sum(data['anticipatory_deflections']) / num_actions
    
    count_recovered, _, count_crashed, _, count_followed = num_helpful_suggestions(data, alone=True)

    return pd.DataFrame( [{
        "perc_destab_actions": destabilizing_prop*100,
        "perc_anticip_actions": prop_anticipatory_deflections * 100,
        "num_crashes": n_crashes,
        "average_crash_prob": avg_crash_probability,
        "crash_freqs": crash_freq,
        "mean_dist_dob": avg_dob_dist,
        "ang_pos_sd": sd_angular_pos,
        "vel_mag_mean": avg_angvel_mag,
        "ang_vel_sd": sd_angvel,
        "vel_rms": angvel_rms,
        "defl_mag_mean": avg_defl_mag,
        "perc_crash_cond_triggered": crash_cond_triggered_prop*100,
        "count_recovered": count_recovered,
        "count_crashed": count_crashed
    }])
            

def process_and_eval_data(study_name, ppt_id, model_id, retrained=False):
    pth_to_data = f"../output/{study_name}/{ppt_id}/{ppt_id}_{model_id}_rt_alone/*/*.csv" if not retrained else f"../output/{study_name}/{ppt_id}/{ppt_id}_{model_id}_rt_retrained_alone/*/*.csv"
    last_run = sorted(glob.glob(pth_to_data), key=os.path.getmtime)[-1]

    data = pd.read_csv(last_run)
    data_stats = get_data_stats(data)
    score = learning_effect(data_stats)
    return score





class RetrainDataset(Dataset):
    def __init__(self, X, Y) -> None:
        super().__init__()
        if not isinstance(X, torch.Tensor):
            self.X = torch.from_numpy(X)
            self.Y = torch.from_numpy(Y)
        else:
            self.X = X
            self.Y = Y

    def __len__(self,):
        return len(self.Y)
    
    def __getitem__(self, idx):
        return self.X[idx].float(), self.Y[idx].float()


# for SAC, DDPG vanilla
class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions

    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)

def pretrain_agent(
    student,
    env,
    train_expert_dataset, test_expert_dataset,
    trace=[],
    batch_size=64,
    epochs=1000,
    scheduler_gamma=0.7,
    learning_rate=1.0,
    log_interval=100,
    no_cuda=False,
    seed=1,
    test_batch_size=64,
):
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if isinstance(env.action_space, gym.spaces.Box):
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Extract initial policy
    model = student.policy.to(device)

    def train(model, device, train_loader, optimizer):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            if isinstance(env.action_space, gym.spaces.Box):
                # A2C/PPO policy outputs actions, values, log_prob
                # SAC/TD3 policy outputs actions only
                if isinstance(student, (A2C, PPO)):
                    action, _, _ = model(data)
                else:
                    # SAC/TD3:
                    action = model(data)
                action_prediction = action.double()
            else:
                # Retrieve the logits for A2C/PPO when using discrete actions
                dist = model.get_distribution(data)
                action_prediction = dist.distribution.logits
                target = target.long()

            loss = criterion(action_prediction, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % log_interval == 0:
                trace.append([epoch, loss.item()])
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                if isinstance(env.action_space, gym.spaces.Box):
                    # A2C/PPO policy outputs actions, values, log_prob
                    # SAC/TD3 policy outputs actions only
                    if isinstance(student, (A2C, PPO)):
                        action, _, _ = model(data)
                    else:
                        # SAC/TD3:
                        action = model(data)
                    action_prediction = action.double()
                else:
                    # Retrieve the logits for A2C/PPO when using discrete actions
                    dist = model.get_distribution(data)
                    action_prediction = dist.distribution.logits
                    target = target.long()

                test_loss = criterion(action_prediction, target)
        test_loss /= len(test_loader.dataset)
        print(f"Test set: Average loss: {test_loss:.4f}")

    # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
    # and testing
    train_loader = torch.utils.data.DataLoader(
        dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_expert_dataset,
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs,
    )

    # Define an Optimizer and a learning rate schedule.
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Now we are finally ready to train the policy model.
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()

    # Implant the trained policy network back into the RL student agent
    student.policy = model

    return student


def retrain_vanilla_rl_model(data, assistant_model_details, experiment_name):

    model_dis_episodes = data

    features, model_actions, human_actions = model_dis_episodes[:, :-2], model_dis_episodes[:, -2], model_dis_episodes[:, -1]

    targets = -1 * model_actions.reshape(-1, 1)


    expert_observations = features
    expert_actions = targets


    env = penalizeDeflection(render_mode='human')

    expert_dataset = ExpertDataSet(expert_observations, expert_actions)
    train_size = int(0.9 * len(expert_dataset))
    test_size = len(expert_dataset) - train_size
    train_expert_dataset, test_expert_dataset = random_split(
        expert_dataset, [train_size, test_size]
    )

    if assistant_model_details['type'] == 'ddpg':
        student_model = DDPG.load(assistant_model_details["path"])
    else:
        student_model = SAC.load(assistant_model_details["path"])
    # env.
    trace = []
    epochs = 100
    lr = 1e-5
    updated_model = pretrain_agent(
        student_model,
        env,
        train_expert_dataset, test_expert_dataset,
        trace=trace,
        epochs=epochs,
        scheduler_gamma=0.7,
        learning_rate=lr,
        log_interval=100,
        no_cuda=False,
        seed=1,
        batch_size=64,
        test_batch_size=10,
    )

    # mean_reward, std_reward = evaluate_policy(updated_model, env, n_eval_episodes=5)

    env.close()

    retrained_path = f"../working_models/assistants/{assistant_model_details['name']}_{experiment_name}_retrained"
    updated_model.save(retrained_path)

    return retrained_path


def retrain_airl_model(data, assistant_model_details, experiment_name):
    model_dis_episodes = data

    features, model_actions, human_actions = model_dis_episodes[:, :-2], model_dis_episodes[:, -2], model_dis_episodes[:, -1]

    targets = -1 * model_actions.reshape(-1, 1)


    env = PendulumEnv()
    # curr_observations = []
    next_observations = []
    infos = []
    dones = []

    for i in range(len(features)):
        env.reset()
        obs = features[i]
        obs = [np.arctan2(obs[1], obs[0]), obs[2]]
        # curr_observations.append(obs)
        action = targets[i]
        env.state = obs
        next_obs, _, terminated, info = env.step(action)
        # print(next_obs)
        next_observations.append(next_obs)
        infos.append(info)

        dones.append(terminated)


    env.close()

    demos = Transitions(
        np.array(features, dtype=np.float64), np.array(targets, dtype=np.float64), np.array(infos), np.array(next_observations, dtype=np.float64), np.array(dones)
    )

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    env = PendulumEnv()
    venv = make_vec_env(PendulumEnv, n_envs=8)
    
    
    learner = SAC.load(assistant_model_details["path"])
    learner.env = venv
    learner.action_noise = action_noise
    learner.learning_rate = 1e-5
    epochs = 100

    reward_net = BasicShapedRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
    )


    airl_trainer = AIRL(
        demonstrations=demos,
        demo_batch_size=64,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True
    )

    airl_trainer.train(epochs)

    venv.close()
    env.close()

    retrained_path = f"../working_models/assistants/{assistant_model_details['name']}_{experiment_name}_retrained"
    learner.save(retrained_path)

    return retrained_path


def retrain_dl_model(data, assistant_model_details, experiment_name):
    model_dis_episodes = data

    features, model_actions, human_actions = model_dis_episodes[:, :-2], model_dis_episodes[:, -2], model_dis_episodes[:, -1]

    targets = human_actions.reshape(-1, 1)

    features = torch.from_numpy(features).float()
    targets = torch.from_numpy(targets).float()

    trainX, testX, trainY, testY = train_test_split(features, targets, test_size=0.1, random_state=42)

    epochs = 20
    lr = 1e-7
    model = load_model(assistant_model_details["path"], assistant_model_details['type'])
    model.lr = lr
    name = f"{assistant_model_details['name']}_hitl_retrain_epochs_{epochs}_lr_{lr}"


    if assistant_model_details['type'] in {'lstm', 'gru', 'rnn'}:
        trainX = trainX.reshape(-1, int(assistant_model_details["window_size"] * 50)+1, 3)
        testX = testX.reshape(-1, int(assistant_model_details["window_size"] * 50)+1, 3)

    
    train_loader, test_loader = None, None
    trainX = (trainX - model.Xmeans) / model.Xstds
    trainY = (trainY - model.Tmeans) / model.Tstds

    testX = (testX - model.Xmeans) / model.Xstds
    testY = (testY - model.Tmeans) / model.Tstds


    train_set = RetrainDataset(trainX, trainY)
    test_set = RetrainDataset(testX, testY)
    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=16, num_workers=0)



    model = load_model(assistant_model_details["path"], assistant_model_details['type'])
    model.lr = lr
    name = f"{assistant_model_details['name']}_hitl_retrain_epochs_{epochs}_lr_{lr}"

    checkpoint_callback = ModelCheckpoint(
                            monitor='val_MAE',
                            dirpath=f"./output/{name}/",
                            filename='best_checkpoint',
                            save_top_k=1,
                            mode='min',
                        )

    early_stopping = EarlyStopping(
        monitor='val_MAE',
        min_delta=0.00001,
        patience= 5,
        mode='min'
    )

    logger = TensorBoardLogger('lightning_logs', name=f"{name}")


    trainer = pl.Trainer(
        max_epochs=epochs,
        precision=32,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        devices="auto"
        # strategy='ddp'
    )

    # trainer.
    trainer.fit(model, train_loader, test_loader) 

    retrained_path = f"./output/{name}/best_checkpoint*.ckpt"
    latest_checkpoint = sorted(glob.glob(retrained_path), key=os.path.getmtime)[-1]
    return latest_checkpoint



def retrain_model(study_name, ppt_id, run_mode, model_id, assistant_model_details):
    pth_to_data = f"../output/{study_name}/{ppt_id}/{run_mode}_{model_id}_HITL/*/disagreement_episodes.txt"
    last_run = sorted(glob.glob(pth_to_data), key=os.path.getmtime)[-1]

    data = np.loadtxt(last_run, delimiter=",")

    if len(data): 
        if data.ndim != 2: 
            data = data.reshape(-1, data.size)
    else:
        print("No disagreement data found")
        return None
    
    if assistant_model_details["type"] in ["ddpg", "sac"] and "airl" not in assistant_model_details["name"]:
        retrained_path = retrain_vanilla_rl_model(data, assistant_model_details, f"{study_name}_{ppt_id}_{model_id}_HITL")
    elif "airl" in assistant_model_details["name"]:
        retrained_path = retrain_airl_model(data, assistant_model_details, f"{study_name}_{ppt_id}_{model_id}_HITL")
    elif assistant_model_details["type"] in ["lstm", "gru", "rnn", "mlp"]:
        retrained_path = retrain_dl_model(data, assistant_model_details, f"{study_name}_{ppt_id}_{model_id}_HITL")
    else:
        print("Model type not supported for retraining")
        retrained_path = None
    

    return retrained_path


if __name__ == "__main__":
    pass
    