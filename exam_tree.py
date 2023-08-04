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
        print("\nresume searching,", agent.ITERATION, "iterations completed before")
        print("=====>loads:", len(agent.nodes), "nodes")
        print("=====>loads:", len(agent.samples), "samples")
        print("=====>loads:", len(agent.DISPATCHED_JOB), "dispatched jobs")
        print("=====>loads:", len(agent.TASK_QUEUE), "task_queue jobs from node:", agent.sample_nodes[0])
        
        print("\nclear the data in nodes...")
        agent.reset_node_data()
        print("finished")

        print("\npopulate prediction data...")
        agent.populate_prediction_data()
        print("finished")

        print("\npredict and partition nets in search space...")
        agent.predict_nodes()
        agent.check_leaf_bags()
        print("finished")
        agent.print_tree()
