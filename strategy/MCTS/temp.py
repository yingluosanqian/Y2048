

from strategy.MCTS.mcts import ReplayBuffer
from strategy.MCTS.network import PolicyValueNet
from const import const_device as device
from const import const_boarder_size as boarder_size
from strategy.MCTS.train import train_nn
from const import const_action as ACTION
from strategy.MCTS.mcts import Strategy
from const import const_device_cpu as device_cpu

import torch
import numpy as np


def train_by_history():
  replay_buffer = ReplayBuffer(max_size=16384)
  for i in range(1, 10 + 1):
    sub_replay_buffer = ReplayBuffer(max_size=1024)
    sub_replay_buffer.load(f"strategy/MCTS/datas/data_{i}.txt")
    replay_buffer.extend(sub_replay_buffer)
  replay_buffer.render()
  print(f"replay_buffer size: {len(replay_buffer)}")

  train_network = PolicyValueNet(
    boarder_size, boarder_size).to(device)  # neural network
  train_nn(
    network=train_network,
    replay_buffer=replay_buffer,
    batch_size=128,
    epochs=50,
    lr=1e-3,
    weight_decay=1e-4
  )

  torch.save(train_network.state_dict(),
             f'strategy/MCTS/models/network_history_{boarder_size}.pth')


def eval_model_once():
  infer_network = PolicyValueNet(
    boarder_size, boarder_size).to(device)
  state_dict = torch.load(f'strategy/MCTS/models/network_history_size_{boarder_size}.pth',
                          map_location=device,
                          weights_only=True)
  infer_network.load_state_dict(state_dict)
  
  strategy = Strategy()
  scores = strategy.collect_trajectory(
    None, network=infer_network, gui=True, collect=False, device=device_cpu)
  print(f"Scores after training: {scores:.2f}")


def eval_model():
  infer_network = PolicyValueNet(
    boarder_size, boarder_size).to(device)
  state_dict = torch.load(f'strategy/MCTS/models/network_history_size_{boarder_size}.pth',
                          map_location=device,
                          weights_only=True)
  infer_network.load_state_dict(state_dict)
  avg_scores = collect_eval_data(infer_network)
  print(f"Average scores after training: {avg_scores:.2f}")


def model_test_by_case():
  chess_board = [
    [64, 128, 2],
    [32, 8, 4],
    [2, 16, 4],
  ]
  state = transform_state(chess_board)
  state = torch.tensor(state, dtype=torch.float64, device=device)
  state = state.unsqueeze(0)  # Add batch dimension if not present

  infer_network = PolicyValueNet(
    boarder_size, boarder_size).to(device)
  state_dict = torch.load(f'strategy/MCTS/models/network_history_{boarder_size}.pth',
                          map_location=device,
                          weights_only=True)
  infer_network.load_state_dict(state_dict)
  infer_network.eval()  # Set to evaluation mode

  policy, _ = infer_network(state)
  policy = policy.detach().squeeze(0)
  policy_view = {}
  for i in range(4):
    policy_view[ACTION[i]] = float(np.round(policy[i], 2))
  print(f"Policy: {policy_view}")


def view_table():
  replay_buffer = ReplayBuffer(max_size=2048)
  replay_buffer.load("strategy/MCTS/datas/data_2.txt")
  replay_buffer.render()


def main():
  # train_by_history()
  # eval_model()
  eval_model_once()
  # model_test_by_case()
  # view_table()


if __name__ == "__main__":
  main()
