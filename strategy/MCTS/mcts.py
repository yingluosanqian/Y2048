
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
from strategy.MCTS.network import PolicyValueNet, encode_state

from const import const_device as device
from const import const_boarder_size as boarder_size
from const import const_action as ACTION
from const import const_fix_action_order as fix_action_order

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

###############################################################################
# Tree Node
###############################################################################


class Tree:
  def __init__(
    self,
    *,
    parent,
    env: Env2048,
    role: Role,
    probability: float,
    brother_id: int,
    moved: bool,
  ):
    self.role: Role = role
    self.children: list[Tree] | None = None
    self.parent: Tree = parent  # parent node
    self.brother_id: int = brother_id  # i-th son of the parent node
    self.env: Env2048 = env
    self.num_of_visit: int = 0  # Number of visits to this node
    self.Q: float = 0  # Q value of the node
    self.probability: float = probability  # Probability of this node being selected
    self.moved: bool = moved  # Whether move happened from parent to this node

  def update(self, Q: float, num_of_visit: int):
    """
    Update the Q value and number of visits for this node.
    """
    self.Q = (self.Q * self.num_of_visit + Q) / (self.num_of_visit + 1)
    self.num_of_visit += num_of_visit

###############################################################################
# MCT
###############################################################################


class MCT:
  def __init__(
    self,
    *,
    baseline_Q: float,
    select_times: int,
    p_v_network: PolicyValueNet,
    device=device,
  ):
    self.baseline_Q = baseline_Q  # Baseline Q value for the root node
    self.select_times = select_times  # Number of times to select a leaf node
    self.p_v_network = p_v_network  # Policy-Value Network
    self.device = device  # Device to run the network
    return

  ###############################################################################
  # Part I: Selection
  ###############################################################################

  def select_leaf(self, cur: Tree) -> Tree:
    """
    Select a leaf node from the current tree using UCT (Upper Confidence Bound for Trees).
    """
    # Extract the UCB score for each child node
    def UCB_score(child: Tree, N: int) -> float:
      return child.Q + UCT_CONSTANT * child.probability * \
          math.sqrt(N) / (1 + child.num_of_visit)

    if cur is None:
      raise ValueError("The current tree is None.")
    if cur.role == Role.GNERATOR:
      # If the current node is a generator, return the first child
      return cur.children[0]
    elif cur.role == Role.PLAYER:
      if cur.children is None:
        return cur
      N = sum(child.num_of_visit for child in cur.children)
      UCT_v = [UCB_score(child, N) for child in cur.children]
      # Sort the 4 children according to UCT_v in descending order
      sorted_indices = np.argsort(UCT_v)[::-1]
      for i in sorted_indices:
        if not cur.children[i].moved:
          continue
        return self.select_leaf(cur.children[i])
      return None
    else:
      raise ValueError("Invalid role for the current tree node.")

  ###############################################################################
  # Part II & III: Expansion & Evaluation
  ###############################################################################

  def expand_and_evaluate(self, cur: Tree):
    if cur.role == Role.PLAYER:
      state = encode_state(cur.env.observation_space.matrix).to(self.device)
      policy, Q_value = self.p_v_network(state)
      policy = policy[0]
      cur.update(Q=Q_value.item(), num_of_visit=1)
      cur.children = []
      for i in range(4):
        env = copy.deepcopy(cur.env)
        moved, _ = env._merge_all(i)
        cur.children.append(
          Tree(
            parent=cur,
            env=env,
            role=Role.GNERATOR,
            probability=policy[i].item(),
            brother_id=i,
            moved=moved,  # Whether the node has moved
          )
        )
        self.expand_and_evaluate(cur.children[i])
    elif cur.role == Role.GNERATOR:
      env = copy.deepcopy(cur.env)
      env._add_new_tile()
      cur.children = []
      cur.children.append(
        Tree(
          parent=cur,
          env=env,
          role=Role.PLAYER,
          probability=1.0,  # Generator always has a single child
          brother_id=0,
          moved=True,  # Always True
        )
      )
    else:
      raise ValueError("Invalid role for the current tree node.")

  ###############################################################################
  # Part IV: Backpropagation
  ###############################################################################

  def backpropagation(self, cur: Tree, N, Q):
    if cur is None:
      return
    cur.update(Q=Q, num_of_visit=N)
    self.backpropagation(cur.parent, N, Q)

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
  ):
    for iter in range(self.select_times):
      # Step 1: Select a leaf node
      expanded_leaf_node = self.select_leaf(root)
      if expanded_leaf_node is None:
        # If no valid leaf node is found, break the loop
        break
      # Step 2 & 3: Expand and evaluate the leaf node
      self.expand_and_evaluate(expanded_leaf_node)
      # Step 4: Backpropagate the score to the root node
      self.backpropagation(expanded_leaf_node.parent, 1, expanded_leaf_node.Q)

    # Return policy
    policy = np.array(
      [child.num_of_visit for child in root.children])
    policy = policy / policy.sum()
    value = root.Q
    return policy, value


class ReplayBuffer(Dataset):
  def __init__(self, max_size=4096):
    self.max_size = max_size
    self.human_states = collections.deque()
    self.states = collections.deque()
    self.policy_targets = collections.deque()
    self.value_targets = collections.deque()

  def add(self, state, policy_target, value_target):
    self.human_states.append(state)
    self.states.append(encode_state(state))
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
    with open("logs/replay_buffer.log", "w") as f:
      f.write("Replay Buffer Contents:\n")
      for i, (state, policy_target, value_target) in enumerate(zip(self.human_states, self.policy_targets, self.value_targets)):
        f.write(f"The {i}-item:\n")
        for row in state:
          f.write(" ".join(f"{num:>{5}}" for num in row) + "\n")
        policy_view = {}
        for i in range(4):
          policy_view[ACTION[i]] = float(np.round(policy_target[i], 2))
        f.write(
          f"Policy_target: {policy_view}, Value_target: {value_target}\n\n")

  def save(self, filename):
    """
    Save the replay buffer to a file.
    """
    with open(filename, "wb") as f:
      torch.save({
        'human_states': self.human_states,
        'states': self.states,
        'policy_targets': self.policy_targets,
        'value_targets': self.value_targets
      }, f)

  def load(self, filename):
    """
    Load the replay buffer from a file.
    """
    with open(filename, "rb") as f:
      data = torch.load(f, weights_only=False)
      self.human_states = data['human_states']
      self.states = data['states']
      self.policy_targets = data['policy_targets']
      self.value_targets = data['value_targets']


class Strategy:
  def __init__(
    self,
    *,
    select_times,
    baseline_score: float,
    p_v_network: PolicyValueNet,
    device=device,
  ):
    self.baseline_score = baseline_score  # Baseline Q value for the root node
    self.select_times = select_times  # Number of times to select a leaf node
    self.p_v_network = p_v_network  # Policy-Value Network
    if self.p_v_network is not None:
      self.p_v_network.eval()
    self.device = device  # Device to run the network
    return

  def take_action(self, env):
    mct = MCT(
      baseline_Q=self.baseline_score,
      select_times=self.select_times,
      p_v_network=self.p_v_network,
      device=self.device
    )
    root = Tree(
      parent=None,
      env=copy.deepcopy(env),
      role=Role.PLAYER,
      probability=0,
      brother_id=0,
      moved=True,
    )
    policy, value = mct.mct_search(root)

    return policy, np.argmax(policy), value

  def collect_trajectory(self, replay_buffer: ReplayBuffer, *, gui=False, collect=True, sleep_time=0.01):
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
      policy, action, value = self.take_action(env)
      origin_state = copy.deepcopy(env.observation_space.matrix)
      _, reward, done, info = env.step(action)
      scores += reward
      if info.moved is False:
        print("[BUG] NOT MOVED !!!!!!!!!!!!!!")
      if collect:
        if not done:
          replay_buffer.add(origin_state, policy, value)
        else:
          value = 1 if scores > self.baseline_score else -1
          replay_buffer.add(origin_state, policy, value)
      if gui:
        game_ui.score += reward
        game_ui.update_grid_cells()
        game_ui.update_idletasks()
        game_ui.update()
        time.sleep(sleep_time)
      if done:
        break
    return scores


def main():
  # eval()
  # score, move = play()
  # print(f"Final Score: {score}")

  replay_buffer = ReplayBuffer()

  p_v_network = PolicyValueNet(
    boarder_size, boarder_size).to(device)  # neural network

  strategy = Strategy(baseline_score=200.0,
                      p_v_network=p_v_network, device=device)
  score = strategy.collect_trajectory(
    replay_buffer, gui=False, sleep_time=0.01)
  print(f"Final Score: {score}")
  replay_buffer.render()


if __name__ == "__main__":
  main()
