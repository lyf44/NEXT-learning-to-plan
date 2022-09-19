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

from env.maze_2d import Maze2D
import utils
from NEXT.model import Model
from NEXT.algorithm import NEXT_plan, RRTS_plan
from NEXT.environment.maze_env import MyMazeEnv

CUR_DIR = osp.dirname(osp.abspath(__file__))
ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)

# def gaussian_probability(sigma, mu, target):
#     """Returns the probability of `target` given MoG parameters `sigma` and `mu`.
#     Arguments:
#         sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
#             size, G is the number of Gaussians, and O is the number of
#             dimensions per Gaussian.
#         mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
#             number of Gaussians, and O is the number of dimensions per Gaussian.
#         target (BxI): A batch of target. B is the batch size and I is the number of
#             input dimensions.
#     Returns:
#         probabilities (BxG): The probability of each point in the probability
#             of the distribution in the corresponding sigma/mu index.
#     """
#     target = target.unsqueeze(1).expand_as(sigma)
#     ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
#     return torch.prod(ret, 2)

def policy_loss(sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """

    m = MultivariateNormal(mu, torch.diag(sigma))
    prob = m.log_prob(target)

    # prob = gaussian_probability(sigma, mu, target)
    # nll = -torch.log(torch.sum(prob, dim=1))
    return -prob

def value_loss(pred, target):
    mse_loss = torch.nn.MSELoss()
    loss = mse_loss(pred, target)
    return loss

def extract_path(search_tree):
    leaf_id = search_tree.states.shape[0]-1

    path = [search_tree.states[leaf_id]]
    id = leaf_id
    while id:
        parent_id = search_tree.rewired_parents[id]
        if parent_id:
            path.append(search_tree.states[parent_id])
        id = parent_id

    path.reverse()

    return path


class MyDataset(Dataset):
    def __init__(self, dataset_size, transform=None, target_transform=None, device="cpu"):
        print("Instantiating dataset...")
        self.transform = transform
        self.target_transform = target_transform
        self.device = device
        self.dataset_size = dataset_size
        # self.dataset = self.load_dataset_from_file()

        # print("dataset size = {}".format(len(self.dataset)))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        file_path = osp.join(data_dir, "data_{}.pkl".format(idx))
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # low = torch.Tensor([-2, -2, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi]).view(1, -1)
        # high = torch.Tensor([2, 2, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi]).view(1, -1)

        occ_grid, start, goal, pos, next_pos, dist_to_g = data

        start_t = torch.Tensor(start)
        goal_t = torch.Tensor(goal)
        occ_grid_t = torch.Tensor(occ_grid).view(1, occ_grid_dim, occ_grid_dim)
        pos_t = torch.Tensor(pos).view(1, robot_dim)
        next_pos_t = torch.Tensor(next_pos).view(-1, robot_dim)
        dist_to_g_t = torch.Tensor([dist_to_g])

        return occ_grid_t, start_t, goal_t, pos_t, next_pos_t, dist_to_g_t

writer = SummaryWriter(comment = '_next')

# Constatns
data_dir = osp.join(CUR_DIR, "dataset")
maze_dir = osp.join(CUR_DIR, "../dataset/gibson/train")

model_path = osp.join(CUR_DIR, "models/next.pt")

# Hyperparameters:
visualize = False
cuda = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch_num = 5000
train_num = 1
UCB_type = 'kde'
robot_dim = 8
bs = 128
occ_grid_dim = 330
train_step_cnt = 5000
sigma = torch.tensor([0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).to(device)

env = MyMazeEnv(robot_dim, maze_dir)

model = Model(cuda = cuda, dim = robot_dim, env_width=occ_grid_dim)
data_cnt = 0
train_data_cnt = train_step_cnt
batch_num = 0
for epoch in range(epoch_num):
    model.net.eval()
    problem = env.init_new_problem()
    model.set_problem(problem)

    if epoch < 1000:
        g_explore_eps = 1.0
    elif epoch < 2000:
        g_explore_eps = 0.5 - 0.1 * (epoch - 1000) / 200
    else:
        g_explore_eps = 0.1

    # Get path
    print("Planning...")
    path = None
    if g_explore_eps == 1.0:
        path = env.expert_path
    else:
        search_tree, done = NEXT_plan(
            env = env,
            model = model,
            T = 500,
            g_explore_eps = g_explore_eps,
            stop_when_success = True,
            UCB_type = UCB_type
        )
        if done:
            path = extract_path(search_tree)

    if path is not None:
        print("Get path, saving to data")

        if visualize:
            utils.visualize_nodes_global(env.map, path, env.init_state, env.goal_state, show=False, save=True, file_name=osp.join(CUR_DIR, "tmp.png"))

        tmp_dataset = []
        for idx in range(1, len(path)):
            pos = path[idx - 1]
            next_pos = path[idx]
            dist_to_g = utils.cal_path_len(path[idx - 1:])
            tmp_dataset.append([env.map, env.init_state, env.goal_state, pos, next_pos, dist_to_g])

        for idx, data in enumerate(tmp_dataset):
            file_path = osp.join(data_dir, "data_{}.pkl".format(data_cnt + idx))
            with open(file_path, 'wb') as f:
                # print("Dumping to {}".format(file_path))
                pickle.dump(tmp_dataset[idx], f)
        data_cnt += len(tmp_dataset)

    print("data_cnt: {}".format(data_cnt))
    writer.add_scalar('dataset_size', data_cnt, epoch)

    if data_cnt > train_data_cnt:
        model.net.train()
        dataset = MyDataset(data_cnt, None, None)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=10, pin_memory=True)
        dataloader_it = iter(dataloader)
        for j in range(train_num):
            occ_grid, start, goal, pos, next_pos, dist_to_g = next(dataloader_it)

            start = start.to(device)
            goal = goal.to(device)
            occ_grid = occ_grid.to(device)
            pos = pos.to(device)
            next_pos = next_pos.to(device)
            dist_to_g = dist_to_g.to(device)

            problem = {
                "map": occ_grid,
                "init_state": start,
                "goal_state": goal
            }

            model.set_problem(problem)
            y = model.net.state_forward(pos, model.pb_rep)
            mu = y[:, :robot_dim]
            v = y[:, -1]

            p_loss = policy_loss(sigma, mu, next_pos)
            v_loss = value_loss(v, dist_to_g)

            loss = p_loss + v_loss

            loss.backward()

            print('Loss after epoch %d, batch_num %d, dataset_sizes %d: , p_loss: %.3f, v_loss: %.3f' % (epoch, batch_num, data_cnt, p_loss.item(), v_loss.item()))
            writer.add_scalar('p_loss/train', p_loss.item(), batch_num)
            writer.add_scalar('v_loss/train', v_loss.item(), batch_num)

            batch_num += 1

            torch.save(model.state_dict(), model_path)
            print("saved session to ", model_path)

        train_data_cnt += train_step_cnt


