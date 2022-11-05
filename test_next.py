import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import torch
import math
import os.path as osp
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
import torch.multiprocessing as mp

# from env.maze_2d import Maze2D
import utils
from NEXT.model import Model
from NEXT.algorithm import NEXT_plan, RRTS_plan
from NEXT.environment.maze_env import MyMazeEnv

CUR_DIR = osp.dirname(osp.abspath(__file__))

def extract_path(search_tree):
    leaf_id = search_tree.states.shape[0]-1

    path = [search_tree.states[leaf_id]]
    id = leaf_id
    while id:
        parent_id = search_tree.rewired_parents[id]
        if parent_id:
            path.append(search_tree.states[parent_id])

        id = parent_id

    path.append(search_tree.non_terminal_states[0]) # append the init state
    path.reverse()

    return path

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name', default='1')
parser.add_argument('--checkpoint', default='')
args = parser.parse_args()

writer = SummaryWriter(comment = '_next')

# Constatns
data_dir = osp.join(CUR_DIR, "dataset")
maze_dir = osp.join(CUR_DIR, "../dataset/gibson/train")

model_path = osp.join(CUR_DIR, "models/next_v2.pt")
best_model_path = osp.join(CUR_DIR, "models/next_v2_best.pt")

# Hyperparameters:
visualize = False
cuda = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_num = 1
train_num = 10
UCB_type = 'kde'
robot_dim = 8
bs = 256
occ_grid_dim = 100
train_step_cnt = 2000
lr = 0.001
alpha_p = 1
alpha_v = 1
start_epoch = 0
sigma = torch.tensor([0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).to(device)

env = MyMazeEnv(robot_dim, maze_dir)
model = Model(env, cuda = cuda, dim = robot_dim, env_width=occ_grid_dim)
mse_loss = torch.nn.MSELoss()

if args.checkpoint != '':
    print("Loading checkpoint {}.pt".format(args.checkpoint))
    model.net.load_state_dict(torch.load(osp.join(CUR_DIR, 'models/{}.pt'.format(args.checkpoint))))

batch_num = 0
best_loss = float('inf')
success_rate = []

for epoch in range(test_num):
    model.net.eval()
    problem = env.init_new_problem()
    model.set_problem(problem)

    g_explore_eps = 0.0

    # Get path
    print("Planning... with explore_eps: {}".format(g_explore_eps))
    path = None

    search_tree, done = NEXT_plan(
        env = env,
        model = model,
        T = 10,
        g_explore_eps = g_explore_eps,
        stop_when_success = True,
        UCB_type = UCB_type
    )
    if done:
        success_rate.append(1)
        path = extract_path(search_tree)

    if path is not None:
        print("Get path, saving to data")
        print(path[0], env.init_state, path[-1], env.goal_state)
        assert np.allclose(np.array(path[0]), np.array(env.init_state))
        assert np.allclose(np.array(path[-1]), np.array(env.goal_state))
        path_tmp = utils.interpolate(path)
        utils.visualize_nodes_global(env.map, path_tmp, env.init_state, env.goal_state, show=False, save=True, file_name=osp.join(CUR_DIR, "res/next_path.png"))
    else:
        path = env.expert_path
        path_tmp = utils.interpolate(path)
        utils.visualize_nodes_global(env.map, path_tmp, env.init_state, env.goal_state, show=False, save=True, file_name=osp.join(CUR_DIR, "res/gt.png"))


