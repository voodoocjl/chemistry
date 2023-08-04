import pickle
import os
import json
import torch
from MCTS_chemisty import MCTS

if __name__ == '__main__':
    # set random seed
    state_path = 'results'
    files = os.listdir(state_path)
    if files:
        files.sort(key=lambda x: os.path.getmtime(os.path.join(state_path, x)))
        node_path = os.path.join(state_path, files[-1])
        # node_path = 'results/mcts_agent_4000'
        with open(node_path, 'rb') as json_data:
            agent = pickle.load(json_data)              
        agent.print_tree()
