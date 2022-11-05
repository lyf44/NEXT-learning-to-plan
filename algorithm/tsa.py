import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import torch
import numpy as np
from queue import Queue
from .search_tree import SearchTree, insert_new_state, compute_w, rewire_to, \
                        set_cost, update_collision_checks
# from .alg_config import RRT_EPS

import utils

CUR_DIR = osp.dirname(osp.abspath(__file__))
VISUALIZE = True

def RRTS_plan(env, T=100, stop_when_success=False):
    return NEXT_plan(env=env, T=T, g_explore_eps=1., \
                    stop_when_success=stop_when_success)

def NEXT_plan(env, model=None, T=100, g_explore_eps=1., \
            stop_when_success=False, UCB_type='kde'):
    """Robot motion planning with NEXT.

    Args:
        env: The environment which stores the problem relevant information (map,
            initial state, goal state), and performs collision check, goal
            region check, uniform sampling.
        model: Machine learning model used to guide vertex selection and
            tree expansion.
        T (int): Maximum number of samples allowed.
        g_explore_eps (float): Probability for RRT-like global exploration.
        stop_when_success (bool): Whether to terminate the algorithm if one path
            is found.
        UCB_type (string): Type of UCB used (one of {'kde', 'GP'}).

    Returns:
        search_tree: Search tree generated by the algorithm.
        success (bool): Whether a path is found.
    """
    search_tree = SearchTree(
        env = env,
        root = env.init_state,
        model = model,
        dim = env.dim
    )

    success = False
    for i in range(T):
        leaf_id = None

        # Goal-biased heuristic.
        if np.random.rand() < 0.1:
            leaf_state, parent_idx, _, no_collision, done = \
                global_explore(search_tree, env, sample_state=env.goal_state)
            success = success or done
            expanded_by_rrt = True

        # RRT-like global exploration.
        elif np.random.rand() < g_explore_eps:
            leaf_state, parent_idx, _, no_collision, done = \
                global_explore(search_tree, env)
            success = success or done
            expanded_by_rrt = True

        # Guided selection and expansion.
        else:
            idx = select(search_tree, env)
            assert search_tree.freesp[idx]
            # assert not search_tree.in_goal_region[idx]

            parent_idx = idx
            leaf_state, _, no_collision, done = \
                expand(search_tree, parent_idx, model, env, extend_cnt=i)
            success = success or done
            expanded_by_rrt = False

        # print(leaf_state)
        leaf_id = insert_new_state(env, search_tree, leaf_state, model, \
            parent_idx, no_collision, done, expanded_by_rrt=expanded_by_rrt)
        RRTS_rewire_last(env, search_tree)

        if VISUALIZE:
            utils.visualize_nodes_global(env.map, search_tree.non_terminal_states, env.init_state, env.goal_state, show=False, save=True, file_name=osp.join(CUR_DIR, "../res/next_tree_{}.png".format(i)))

        if success and stop_when_success:
            break

    print('success =', success, '   number of samples =', i)

    return search_tree, success

def RRT_steer(env, sample_state, nearest, dist):
    """Steer the sampled state to a new state close to the search tree.

    Args:
        env: The environment which stores the problem relevant information (map,
            initial state, goal state), and performs collision check, goal
            region check, uniform sampling.
        sample_state: State sampled from some distribution.
        nearest: Nearest point in the search tree to the sampled state.
        dist: Distance between sample_state and nearest.

    Returns:
        new_state: Steered state.
    """
    if dist < env.RRT_EPS:
        return sample_state

    base_dist = np.linalg.norm(nearest[:2] - sample_state[:2])
    ratio = env.RRT_EPS / base_dist
    return env.interpolate(nearest, sample_state, ratio)

def global_explore(search_tree, env, sample_state=None):
    """One step of RRT-like expansion.

    Args:
        search_tree: Current search tree generated by the algorithm.
        env: The environment which stores the problem relevant information (map,
            initial state, goal state), and performs collision check, goal
            region check, uniform sampling.
        sample_state: A randomly sampled state (if provided).

    Returns:
        new_state: New state being added to the search tree.
        parent_idx: Index of the parent of the new state.
        action: Path segment connecting parent and new state.
        no_collision (bool): True <==> the path segment is collision-free.
        done (bool): True <==> the path segment is collision-free and the new
            state is inside the goal region.
    """
    non_terminal_states = search_tree.non_terminal_states

    # Sample uniformly in the maze
    if sample_state is None:
        sample_state = env.uniform_sample()

    # Steer sample to nearby location
    dists = env.distance(non_terminal_states, sample_state)
    nearest_idx, min_dist = np.argmin(dists), np.min(dists)
    new_state = RRT_steer(env, sample_state, non_terminal_states[nearest_idx], \
                        min_dist)
    # new_state = sample_state

    res_new_state, action, no_collision, done = env.step(
        state = non_terminal_states[nearest_idx],
        new_state = new_state
    )

    # print("rrt_extend", non_terminal_states[nearest_idx], sample_state, res_new_state, no_collision, done)

    # if np.allclose(res_new_state, non_terminal_states[nearest_idx]):
    #     res_new_state = new_state

    return res_new_state, search_tree.non_terminal_idxes[nearest_idx], action, \
        no_collision, done

def select(search_tree, env, c=1., use_GP=False):
    """Select a point in the search tree for expansion.

    Args:
        search_tree: Current search tree generated by the algorithm.
        env: The environment which stores the problem relevant information (map,
            initial state, goal state), and performs collision check, goal
            region check, uniform sampling.
        c: Hyperparameter controlling the weight for exploration.
        use_GP: True <==> using Gaussian Process.

    Returns:
        idx (int): Index of the point in the tree being selected.
    """
    scores = []
    for i in range(search_tree.non_terminal_states.shape[0]):
        idx = search_tree.non_terminal_idxes[i]
        Q = search_tree.state_values[idx]
        print(search_tree.w_sum, search_tree.w[idx], idx)
        U = np.sqrt(np.log(search_tree.w_sum) / search_tree.w[idx])

        scores.append(Q + c*U)

    return search_tree.non_terminal_idxes[np.argmax(scores)]

@torch.no_grad()
def expand(search_tree, idx, model, env, k=10, c=1., extend_cnt=0, use_GP=False):
    """Expand a search tree from a given point.

    Args:
        search_tree: Current search tree generated by the algorithm.
        idx (int): Index of the selected point.
        model: Machine learning model used to guide the expansion.
        env: The environment which stores the problem relevant information (map,
            initial state, goal state), and performs collision check, goal
            region check, uniform sampling.
        k (int): Number of candidate actions.
        c: Hyperparameter controlling the weight for exploration.
        use_GP: True <==> using Gaussian Process.

    Returns:
        new_state: New state being added to the tree.
        action: Path segment connecting parent and new state.
        no_collision (bool): True <==> the path segment is collision-free.
        done (bool): True <==> the path segment is collision-free and the new
            state is inside the goal region.
    """
    state = np.array(search_tree.states[idx])
    candidate_actions = model.policy(state=state, k=k)[0]
    candidates = []
    for i in range(k):
        action = candidate_actions[i]
        new_state, _ = env.step(state=state, action=action, \
                                    check_collision=False)
        candidates.append(new_state)

    if k > 1:
        scores = []
        Qs = model.pred_value(np.array(candidates))
        for i in range(k):
            Q = Qs[i]
            w = compute_w(env, search_tree, state=candidates[i])
            U = np.sqrt(np.log(search_tree.w_sum) / w)
            scores.append(Q + c*U)
        new_state = candidates[np.argmax(scores)]

    else:
        new_state = candidates[0]

    target_state = new_state

    if VISUALIZE:
        utils.visualize_nodes_global(env.map, [new_state], state, env.goal_state, show=False, save=True, file_name=osp.join(CUR_DIR, "../res/next_extension_{}.png".format(extend_cnt)))

    new_state, action, no_collision, done = env.step(
        state = state,
        new_state = new_state
    )

    print("neural_extend", no_collision, done)
    return new_state, action, no_collision, done

def RRTS_rewire_last(env, search_tree, neighbor_r=None, obs_cost=2):
    """Locally optimize the search tree by rewiring the latest added point.

    Args:
        env: The environment which stores the problem relevant information (map,
            initial state, goal state), and performs collision check, goal
            region check, uniform sampling.
        search_tree: Current search tree generated by the algorithm.
        neighbor_r (float): Radius for rewiring.
        obs_cost (float): Cost for obstacle (hyperparameter).
    """
    if neighbor_r is None:
        neighbor_r = env.RRT_EPS*3
    cur_tree = search_tree.states[:-1]
    new_state = search_tree.states[-1]
    nearest = search_tree.parents[-1]
    freesp = search_tree.freesp

    # Return if the latest point is inside of an obstacle.
    if not search_tree.freesp[-1]:
        set_cost(search_tree, -1, obs_cost)
        update_collision_checks(search_tree, env.collision_check_count)
        return

    # Find the locally optimal path to the root for the latest point.
    dists = env.distance(cur_tree, new_state)
    near = np.where(dists < neighbor_r)[0]

    min_cost = dists[nearest] + search_tree.costs[nearest]
    min_j = nearest
    for j in near:
        if not freesp[j]:
            continue
        cost_new = dists[j] + search_tree.costs[j]
        if cost_new < min_cost:
            _, _, no_collision, done = env.step(
                state = cur_tree[j],
                new_state = new_state
            )
            if no_collision:
                min_cost, min_j = cost_new, j

    # Rewire (change parent) to the locally optimal path.
    rewire_to(search_tree, -1, min_j)
    set_cost(search_tree, -1, min_cost)

    # If the latest point can improve the cost for the neighbors, rewire them.
    for j in near:
        cost_new = min_cost + dists[j]
        if cost_new < search_tree.costs[j]:
            _, _, no_collision, done = env.step(
                state = cur_tree[j],
                new_state = new_state
            )

            if no_collision:
                set_cost(search_tree, j, cost_new)
                rewire_to(search_tree, j, len(search_tree.states)-1)

    update_collision_checks(search_tree, env.collision_check_count)