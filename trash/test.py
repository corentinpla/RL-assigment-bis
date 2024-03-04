with open("/Users/corentinpla/Documents/C-1-GitHub/RL-assigment/opt_params.txt", 'rb') as f:
    # Load the data from the file
    results_dict = pickle.load(f)

# Now you can access the data from the results_dict
best_learning_rate = results_dict['best_learning_rate']
best_gamma = results_dict['best_gamma']
best_exploration_decay = results_dict['best_exploration_decay']
best_dropout_rate = results_dict['best_dropout_rate']
episode_returns = results_dict['episode_returns']

# Do whatever you need to do with the loaded data
print("Best Learning Rate:", best_learning_rate)
print("Best Gamma:", best_gamma)
print("Best Exploration Decay:", best_exploration_decay)
print("Best Dropout Rate:", best_dropout_rate)
print("Episode Returns:", episode_returns)