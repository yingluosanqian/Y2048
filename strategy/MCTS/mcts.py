
import copy
import time
import math
import numpy as np
from functools import lru_cache
import collections
import random
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from game.env_2048 import Env2048
from game.game_2048_ui import Game2048UI
ENV = Env2048  # Alias for the environment
UI = Game2048UI  # Alias for the environment
from strategy.MCTS.network import PolicyValueNet, encode_state

from const import const_device as device
from const import const_boarder_size as boarder_size
from const import const_action as ACTION
from const import const_fix_action_order as fix_action_order

UCT_CONSTANT = 1.5


class Role:
  PLAYER = "player"
  GNERATOR = "generator"


class TreeNode:
  def __init__(
    self,
    *,
    parent,
    action: int,
    probability: float,
  ):
    self.children: list[TreeNode] | None = None
    self.parent: TreeNode = parent  # parent node
    self.action = action
    self.num_of_visit: int = 0  # Number of visits to this node
    self.Q: float = 0  # Q value of the node
    self.probability: float = probability  # Probability of this node being selected

  def compute_UCB_score(self, total_visits: int) -> float:
    return self.Q + UCT_CONSTANT * self.probability * math.sqrt(
      total_visits / (1 + self.num_of_visit))

  def update(self, Q: float, num_of_visit: int):
    """
    Update the Q value and number of visits for this node.
    """
    self.Q = (self.Q * self.num_of_visit + Q) / \
        (self.num_of_visit + num_of_visit)
    self.num_of_visit += num_of_visit


def add_root_dirichlet_noise(priors):
  alpha = 0.3
  noise = np.random.dirichlet([alpha] * len(priors))

  eta = 0.25
  return (1 - eta) * priors + eta * noise
###############################################################################
# MCT
###############################################################################


class MCT:
  def __init__(
    self,
    *,
    env: ENV,
    simulation_times: int,
    p_v_network: PolicyValueNet,
    device,
  ):
    self.env = env
    # Number of times to select a leaf node
    self.simulation_times = simulation_times
    self.p_v_network = p_v_network  # Policy-Value Network
    self.device = device  # Device to run the network
    self.root = TreeNode(
      parent=None,
      action=None,  # Root node has no action
      probability=0,
    )
    return

  ###############################################################################
  # Overview: MCTS Search Algorithm
  # 1. Selection: Collect all selectable child nodes from the root.
  # 2. Expansion: Expand the tree by selecting a child node and creating a new node.
  # 3. Evaluation: Evaluate rather than rollout the game.
  # 4. Backpropagation: Backpropagate the score to the root node.
  ###############################################################################

  def simulate(self):
    ###############################################################################
    # Part I: Selection
    ###############################################################################
    cur = self.root
    env: ENV = copy.deepcopy(self.env)
    while cur.children is not None:
      if cur.children == []:
        break
      UCT_v = [child.compute_UCB_score(cur.num_of_visit)
               for child in cur.children]
      # Select the child with the highest UCT score
      child_id = np.argmax(UCT_v)
      env.step(cur.children[child_id].action)
      cur = cur.children[child_id]
    ###############################################################################
    # Part II & III: Expansion & Evaluation
    ###############################################################################
    state = encode_state(env.observation_space.matrix).to(self.device)
    with torch.no_grad():
      self.p_v_network.eval()
      policy, Q_value = self.p_v_network(state)
      policy = F.softmax(policy, dim=1)
      policy, Q_value = policy[0].cpu().numpy(), Q_value[0].item()
      if cur == self.root:
        policy = add_root_dirichlet_noise(policy)

    cur.update(Q=Q_value, num_of_visit=1)
    cur.children = []
    for i in env.available_actions():
      sub_env = copy.deepcopy(env)
      _, _, _, info = sub_env.step(i)
      if info.moved:
        cur.children.append(
          TreeNode(
            parent=cur,
            action=i,
            probability=policy[i],
          )
        )
    ###############################################################################
    # Part IV: Backpropagation
    ###############################################################################
    cur = cur.parent
    while cur is not None:
      cur.update(Q=Q_value, num_of_visit=1)
      cur = cur.parent

  def mct_search(self):
    for iter in range(self.simulation_times):
      self.simulate()
    # Return policy
    policy = np.zeros((self.env.action_space.n))
    for child in self.root.children:
      policy[child.action] = child.num_of_visit

    policy = policy / policy.sum()
    value = self.root.Q
    return policy, value


class ReplayBuffer(Dataset):
  def __init__(self, max_size=2048):
    self.max_size = max_size
    self.human_states = collections.deque()
    self.states = collections.deque()
    self.policy_targets = collections.deque()
    self.value_targets = collections.deque()

  def add(self, human_state, state, policy_target, value_target):
    self.human_states.append(human_state)
    self.states.append(state)
    self.policy_targets.append(policy_target)
    self.value_targets.append(float(value_target))

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
    simulation_times,
    temperature,
    p_v_network: PolicyValueNet,
    device,
  ):
    # Number of times to select a leaf node
    self.simulation_times = simulation_times
    self.temperature = temperature  # Temperature for exploration
    self.p_v_network = p_v_network  # Policy-Value Network
    self.device = device  # Device to run the network
    return

  def take_action(self, env):
    mct = MCT(
      env=copy.deepcopy(env),
      simulation_times=self.simulation_times,
      p_v_network=self.p_v_network,
      device=self.device
    )
    policy, value = mct.mct_search()

    return policy, np.argmax(policy), value

  def play(
    self,
    *,
    collect=True,
    gui=False,
    sleep_time=0.01,
  ):
    """
    Collect a trajectory of actions and rewards from the environment.
    Returns ...
    """
    env = ENV()  # environment

    if gui:
      game_ui = UI(env, player_mode=False)
      game_ui.update_idletasks()
      game_ui.update()

    trajectory = []
    done = False
    score = 0.0
    while not done:
      policy, action, _ = self.take_action(env)

      if random.random() < self.temperature:
        # Randomly select an action
        action = env.sample()

      human_origin_state = copy.deepcopy(env.observation_space.matrix)
      origin_state = encode_state(human_origin_state, unsqueeze=False)

      if collect:
        trajectory.append(
          (human_origin_state, origin_state, policy))

      _, reward, done, info = env.step(action)
      score += reward
      if gui:
        game_ui.update_grid_cells()
        game_ui.update_idletasks()
        game_ui.update()
        time.sleep(sleep_time)
      if done:
        if gui:
          time.sleep(1)
        human_state = copy.deepcopy(env.observation_space.matrix)
        state = encode_state(human_state, unsqueeze=False)
        trajectory.append(
          (human_state, state, np.zeros(policy.shape) / env.action_space.n))
        return trajectory, env._eval_value(), score

  def collect_trajectory(
    self,
    *,
    replay_buffer: ReplayBuffer,
    gui=False,
  ):
    """
    Collect a trajectory of actions and rewards from the environment.
    Returns ...
    """
    trajectory, value, score = self.play(collect=True, gui=gui)

    for human_state, origin_state, policy in reversed(trajectory):
      # Encode the state and add to the replay buffer
      replay_buffer.add(
        human_state=human_state,
        state=origin_state,
        policy_target=policy,
        value_target=value,
      )
    return score


def main():
  # eval()
  # score, move = play()
  # print(f"Final Score: {score}")

  replay_buffer = ReplayBuffer()

  p_v_network = PolicyValueNet(
    boarder_size, boarder_size).to(device)
  state_dict = torch.load(f'strategy/MCTS/models/network_latest_size_{boarder_size}.pth',
                          map_location=device,
                          weights_only=True)
  p_v_network.load_state_dict(state_dict)

  strategy = Strategy(
    simulation_times=20,
    temperature=0,
    p_v_network=p_v_network,
    device=device,
  )
  score = strategy.collect_trajectory(replay_buffer=replay_buffer, gui=True)
  print(f"Final Score: {score}")
  replay_buffer.render()


if __name__ == "__main__":
  main()
