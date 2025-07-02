
import copy
import time
import math
import numpy as np
from functools import lru_cache
import collections
import random
import torch
from torch.utils.data import Dataset

from game.env_2048 import Env2048
from game.game_2048_ui import Game2048UI
from strategy.MCTS.network import PolicyValueNet

device = "cuda" if torch.cuda.is_available() else "cpu"

ACTION = ["Left", "Right", "Up", "Down"]
UCT_CONSTANT = 1.1111


class Role:
  PLAYER = "player"
  GNERATOR = "generator"


###############################################################################
# Utils
###############################################################################
def merge_lists(list1, list2):
  """
  Merge two lists and return a new list with unique elements.
  """
  return list(set(list1) | set(list2))


def transform_state(obs: list[list[int]]) -> np.array:
  n = len(obs)
  m = len(obs[0])
  max_exp = 17
  one_hot = np.zeros((max_exp + 1, n, m), dtype=np.float64)
  for i in range(n):
    for j in range(m):
      val = obs[i][j]
      if val == 0:
        one_hot[max_exp, i, j] = 1.0
      else:
        n = int(np.log2(val))
        if 1 <= n <= max_exp:
          one_hot[n - 1, i, j] = 1.0
  return one_hot

###############################################################################
# Tree Node
###############################################################################


class Tree:
  def __init__(self, *, parent, env: Env2048, role, score_gain):
    self.role = role
    if role == Role.GNERATOR:
      # TODO: Replace the random strategy
      self.childs: list[Tree | None] = [None]
      self.score_gain = score_gain
    elif role == Role.PLAYER:
      self.childs: list[Tree | None] = [None, None, None, None]
    else:
      raise ValueError()
    self.parent = parent
    self.env = env
    # How many score could gain from this node
    self.perf_update_count = 0
    self.perf = 0.0
    # How many times this node has been selected
    self.n = 1

  def is_full_child(self):
    """
    Check if all child nodes are filled.
    Returns True if all child nodes are not None, otherwise False.
    """
    return all(child is not None for child in self.childs)

  def update_perf(self, score):
    """
    Update the performance score of the current node.
    """
    self.perf = (self.perf * self.perf_update_count + score) / \
        (self.perf_update_count + 1)
    self.perf_update_count += 1

  def debug(self):
    print(
      f"Num of non-empty childs: {len([c for c in self.childs if c is not None])}")


###############################################################################
# MCT
###############################################################################
class MCT:
  def __init__(self):
    pass
  ###############################################################################
  # Part I: Selection
  ###############################################################################

  def select_child_order(self, trees: list[Tree], N: int) -> list[Tree]:
    """
    Select a child node from the list of trees.
    """
    if not trees:
      raise ValueError("No selectable child nodes available.")
    max_perf = max(tree.perf for tree in trees) + 0.1
    trees.sort(key=lambda tree: tree.perf / max_perf + UCT_CONSTANT *
               math.sqrt(math.log(N) / tree.n), reverse=True)
    return trees

  def collect_selectable_childs(self, root: Tree):
    """
    Collects all selectable child nodes from the root.
    Returns a list of child nodes that can be selected.
    """
    if not root:
      return []
    if root.role == Role.GNERATOR:
      return self.collect_selectable_childs(root.childs[0])
    else:
      selectable_childs = [root] if not root.is_full_child() else []
      for child in root.childs:
        selectable_childs += self.collect_selectable_childs(child)
      return selectable_childs

  ###############################################################################
  # Part II: Expansion
  ###############################################################################

  def expand_tree(self, cur: Tree):
    if cur.role == Role.PLAYER:
      # TODO: Replace this strategy
      shuffled_index = np.random.permutation(range(4))
      for i in shuffled_index:
        if cur.childs[i] is None:
          env = copy.deepcopy(cur.env)
          moved, score_gain = env._merge_all(i)
          if not moved:
            continue
          cur.childs[i] = Tree(parent=cur, env=env, role=Role.GNERATOR,
                               score_gain=score_gain)
          return self.expand_tree(cur.childs[i])
    else:
      env = copy.deepcopy(cur.env)
      env._add_new_tile()
      cur.childs[0] = Tree(parent=cur, env=env,
                           role=Role.PLAYER, score_gain=0.0)
      return cur.childs[0]
    return None

  ###############################################################################
  # Part III: Simulation
  ###############################################################################

  def simulate(self, cur: Tree, network: PolicyValueNet | None = None):
    env = copy.deepcopy(cur.env)
    state = transform_state(env.observation_space.matrix)
    done = env._is_game_over()
    score = 0
    move_count = 0
    while not done and move_count < 32:
      actions = [network.take_action(state)] if network else []
      actions += list(np.random.permutation(range(4)))
      for i in actions:
        next_state, reward, done, info = env.step(i)
        state = transform_state(next_state)
        if info.moved is False:
          continue
        score += reward
        break
      move_count += 1
    return score

  ###############################################################################
  # Part IV: Backpropagation
  ###############################################################################

  def backpropagation(self, cur: Tree, score):
    if not cur:
      return
    if cur.parent is not None and cur.role == Role.PLAYER:
      score += cur.parent.score_gain
    cur.update_perf(score)
    self.backpropagation(cur.parent, score)

  ###############################################################################
  # Overview: MCTS Search Algorithm
  # 1. Selection: Collect all selectable child nodes from the root.
  # 2. Expansion: Expand the tree by selecting a child node and creating a new node.
  # 3. Simulation: Simulate the game from the expanded tree.
  # 4. Backpropagation: Backpropagate the score to the root node.
  ###############################################################################

  def mct_search(
    self,
    root: Tree,
    *,
    network: PolicyValueNet | None = None,
    select_times,
  ):
    selectable_childs = []
    for iter in range(select_times):
      # Step 1: Select a child node
      selectable_childs = self.collect_selectable_childs(root)
      ordered_selectable_childs = self.select_child_order(
        selectable_childs, iter + 1)
      # Step 2: Expand the tree
      expanded_tree = None
      for selected_tree in ordered_selectable_childs:
        expanded_tree = self.expand_tree(selected_tree)
        if expanded_tree is not None:
          break
      if expanded_tree is None:
        break
      # Step 3: Simulate the game from the expanded tree
      score = self.simulate(expanded_tree, network=network)
      # Step 4: Backpropagate the score to the root
      self.backpropagation(expanded_tree, score)

    # Return policy and value
    policy = np.array(
      [child.perf if child is not None else 0.0 for child in root.childs])
    value = root.perf
    return policy, value


class ReplayBuffer(Dataset):
  def __init__(self, max_size=1024):
    self.max_size = max_size
    self.human_states = collections.deque()
    self.states = collections.deque()
    self.policy_targets = collections.deque()
    self.value_targets = collections.deque()

  def add(self, state, policy_target, value_target):
    self.human_states.append(state)
    self.states.append(transform_state(state))
    self.policy_targets.append(policy_target)
    self.value_targets.append(value_target)

  def extend(self, other):
    """
    Extend the replay buffer with another ReplayBuffer instance.
    """
    if not isinstance(other, ReplayBuffer):
      raise TypeError("Can only extend with another ReplayBuffer instance.")
    self.human_states.extend(other.human_states)
    self.states.extend(other.states)
    self.policy_targets.extend(other.policy_targets)
    self.value_targets.extend(other.value_targets)

    # Ensure the size does not exceed max_size
    while len(self.states) > self.max_size:
      self.human_states.popleft()
      self.states.popleft()
      self.policy_targets.popleft()
      self.value_targets.popleft()

  def __len__(self):
    return len(self.states)

  def __getitem__(self, idx):
    if idx < 0 or idx >= len(self.states):
      raise IndexError("Index out of bounds")
    return self.states[idx], self.policy_targets[idx], self.value_targets[idx]

  def render(self):
    """
    Render the replay buffer contents.
    This is a placeholder function and can be implemented as needed.
    """
    with open("logs/replay_buffer.txt", "w") as f:
      f.write("Replay Buffer Contents:\n")
      for i, (state, policy_target, value_target) in enumerate(zip(self.human_states, self.policy_targets, self.value_targets)):
        f.write(f"The {i}-item:\n")
        for row in state:
          f.write(" ".join(f"{num:>{5}}" for num in row) + "\n")
        f.write(
          f"Policy_target: {np.round(policy_target, 2)}, Value_target: {value_target}\n\n")


class Strategy:
  def __init__(self):
    pass

  def take_action(self, env, *, network=None, select_times=10):
    mct = MCT()
    tree = Tree(parent=None, env=copy.deepcopy(
      env), role=Role.PLAYER, score_gain=0.0)
    policy, value = mct.mct_search(
      tree, network=network, select_times=select_times)
    return policy, value, np.argmax(policy)

  def collect_trajectory(self, replay_buffer: ReplayBuffer, *, network=None, gui=False, collect=True):
    """
    Collect a trajectory of actions and rewards from the environment.
    Returns ...
    """
    env = Env2048()  # environment

    if gui:
      game_ui = Game2048UI(env, player_mode=False)
      game_ui.update_idletasks()
      game_ui.update()

    done = False
    scores = 0
    while not done:
      policy, value, action = self.take_action(env, network=network)
      if action == -1:  # No valid move
        break
      origin_state = copy.deepcopy(env.observation_space.matrix)
      _, reward, done, _ = env.step(action)
      if collect:
        replay_buffer.add(origin_state, policy, scores + value)
      scores += reward
      if gui:
        game_ui.score += reward
        game_ui.update_grid_cells()
        game_ui.update_idletasks()
        game_ui.update()
        time.sleep(0.05)
      if done:
        break
    return scores

def main():
  # eval()
  # score, move = play()
  # print(f"Final Score: {score}")

  replay_buffer = ReplayBuffer()

  # network = PolicyValueNet(3, 3, num_res_blocks=6).to(device)  # neural network
  network = None
  strategy = Strategy()
  strategy.collect_trajectory(replay_buffer, network=network, gui=True)
  replay_buffer.render()


if __name__ == "__main__":
  main()
