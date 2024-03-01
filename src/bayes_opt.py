
import skopt
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
import pickle 

from train import ProjectAgent


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
) 

    
def optimize_hyperparameters(agent):

    def objective(params):
        agent.learning_rate = params[0]
        agent.gamma = params[1]
        agent.exploration_decay = params[2]
        agent.model.dropout= nn.Dropout(params[3])
        

        episode_returns = agent.train(env, agent.max_episode)
        average_return = np.mean(episode_returns)
        
        return -average_return

    search_space = [
        Real(0.00001, 0.1, name='learning_rate'),       
        Real(0, 0.999, name='gamma'),  
        Real (0.8, 0.99, name="exploration_decay"),                  
        Real(0.01, 0.5, name='dropout_rate')            
        ]

    result = gp_minimize(
        objective, search_space, n_calls=40, random_state=42
    )


    best_learning_rate = result.x[0]
    best_gamma = result.x[1]
    best_exploration_decay = result.x[2]
    best_dropout_rate = result.x[3]

    print("Best hyperparameters found:")
    print("Learning Rate:", best_learning_rate)
    print("Gamma:", best_gamma)
    print("Best exploration decay",best_exploration_decay)
    print("Dropout Rate:", best_dropout_rate)


    agent.learning_rate = best_learning_rate
    agent.gamma = best_gamma
    agent.exploration_decay = best_exploration_decay
    agent.model.dropout = nn.Dropout(best_dropout_rate)


    episode_returns = agent.train(env, agent.max_episode)
    print('episode_returns', episode_returns)

    with open("/Users/corentinpla/Documents/C-1-GitHub/RL-assigment/opt_params.txt", 'wb') as f:
        results_dict = {
            'best_learning_rate': best_learning_rate,
            'best_gamma': best_gamma,
            'best_exploration_decay': best_exploration_decay,
            'best_dropout_rate': best_dropout_rate,
            'episode_returns': episode_returns
        }
        pickle.dump(results_dict, f)

    return episode_returns

agent = ProjectAgent()
episode_returns = optimize_hyperparameters(agent)