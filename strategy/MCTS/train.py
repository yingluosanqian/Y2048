
from const import const_device as device
from const import const_device_cpu as device_cpu
from const import const_boarder_size as boarder_size
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import time
from pathlib import Path

from tqdm import tqdm

from strategy.MCTS.network import PolicyValueNet
from strategy.MCTS.mcts import Strategy, ReplayBuffer
from torch.utils.data import Dataset, DataLoader

import logging
import multiprocessing

logging.basicConfig(
  filename='logs/main.log',
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
)


class ReplayBufferDataset(Dataset):
  def __init__(self, replay_buffer):
    self.buffer = replay_buffer

  def __len__(self):
    return len(self.buffer)

  def __getitem__(self, idx):
    sample = self.buffer[idx]
    return np.array(sample['states'], dtype=np.float64), np.array(sample['actions'], dtype=np.float64), np.array(sample['rewards'], dtype=np.float64)

###############################################################################
# Collecting training data from multiple workers
###############################################################################


def _collect_trajectory_worker(args):
  strategy: Strategy = args[0]
  buffer_size = args[1]
  local_buffer = ReplayBuffer(max_size=buffer_size)
  scores = []
  while len(local_buffer) < buffer_size:
    scores.append(strategy.collect_trajectory(local_buffer, gui=False))
  return local_buffer, scores


def collect_train_data(
  network,
  replay_buffer: ReplayBuffer,
  num_workers,
  baseline_score,
  temperature,
):
  # Ensure the network is on CPU for multiprocessing
  network = network.to(device_cpu)
  with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
    # Each worker gets its own Strategy and collects samples
    sub_buffer_size = (replay_buffer.max_size + num_workers - 1) // num_workers
    logging.info(f"Sub buffer size for each worker: {sub_buffer_size}")
    args = []
    for _ in range(num_workers):
      strategy = Strategy(
        temperature=temperature,
        select_times=20,
        baseline_score=baseline_score,
        p_v_network=network,
        device=device_cpu
      )
      args.append((strategy, sub_buffer_size))
    results = pool.map(_collect_trajectory_worker, args)
    # Flatten and add to replay_buffer

    scores = []
    for worker_samples, score_list in results:
      replay_buffer.extend(worker_samples)
      scores += score_list
  return scores


def get_dataloader_from_replaybuffer(replay_buffer, batch_size=128, shuffle=True):
  dataset = ReplayBufferDataset(replay_buffer)
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_nn(
  network: PolicyValueNet,
  replay_buffer: ReplayBuffer,
  epochs=50,
  batch_size=128,
  lr=1e-3,
  weight_decay=1e-4,
):
  """
  Train the neural network using the data from the replay buffer.
  This function is a placeholder and should be implemented with actual training logic.
  """
  dataloader = DataLoader(replay_buffer, batch_size=batch_size, shuffle=False)

  # Loss functions
  def policy_loss(policy_logits, policy_target):
    return F.cross_entropy(policy_logits, policy_target)

  def value_loss(value_pred: torch.Tensor, value_target):
    return F.mse_loss(value_pred.squeeze(), value_target)

  optimizer = optim.Adam(network.parameters(), lr=lr,
                         weight_decay=weight_decay)

  for epoch in range(epochs):
    total_policy_loss, total_value_loss = 0.0, 0.0
    for batch_idx, (inputs, policy_target, value_target) in enumerate(dataloader):
      inputs = inputs.to(device)
      policy_target = policy_target.to(device)
      value_target = value_target.to(device)

      optimizer.zero_grad()

      policy_logits, value_pred = network(inputs)
      loss_policy = policy_loss(policy_logits, policy_target)
      loss_value = value_loss(value_pred, value_target)
      total_loss = loss_policy + loss_value

      total_loss.backward()
      optimizer.step()

      total_policy_loss += loss_policy.item()
      total_value_loss += loss_value.item()
    avg_policy_loss = total_policy_loss / len(dataloader)
    avg_value_loss = total_value_loss / len(dataloader)
    logging.info(
      f"Epoch {epoch+1}: Policy Loss = {avg_policy_loss:.4f}, Value Loss = {avg_value_loss:.4f}")


def main():
  logging.info("====================================================")
  logging.info("Starting training process...")
  logging.info("====================================================")

  ###############################################################################
  # Policy-Value Network Initialization
  ###############################################################################
  p_v_network = PolicyValueNet(
    boarder_size, boarder_size).to(device)  # neural network
  network_path = Path(
    f'strategy/MCTS/models/network_latest_size_{boarder_size}.pth')
  if network_path.exists():
    logging.info("Loading existing network weights...")
    torch.load(f'strategy/MCTS/models/network_latest_size_{boarder_size}.pth',
               p_v_network.state_dict(), weights_only=True)
  else:
    logging.info("No existing network weights found, starting from scratch.")

  ###############################################################################
  # Main training loop
  ###############################################################################
  baseline_score, avg_score, num = 0.0, 0.0, 0
  logging.info(f"Calculate initial baseline score...")
  score_list = collect_train_data(
    network=p_v_network,
    replay_buffer=ReplayBuffer(),
    num_workers=10,
    baseline_score=baseline_score,
    temperature=0
  )
  avg_score = (avg_score * num + sum(score_list)) / (num + len(score_list))
  baseline_score = max(baseline_score, avg_score)
  num += len(score_list)
  logging.info(f"Initial baseline score: {baseline_score:.2f}")
  for i in range(200):
    replay_buffer = ReplayBuffer()

    ###############################################################################
    # Collect training data and train the neural network
    ###############################################################################
    logging.info(f"Collecting training data, iteration {i+1}...")
    score_list = collect_train_data(
      network=p_v_network,
      replay_buffer=replay_buffer,
      num_workers=10,
      baseline_score=baseline_score,
      temperature=0.1 if i < 5 else 0.0,
    )
    avg_score = (avg_score * num + sum(score_list)) / (num + len(score_list))
    baseline_score = max(baseline_score, avg_score)
    num += len(score_list)
    logging.info(f"Baseline score after iteration {i+1}: {baseline_score:.2f}")
    logging.info(f"Average score after iteration {i+1}: {avg_score:.2f}")
    replay_buffer.save(f"strategy/MCTS/datas/data_{i+1}.txt")

    ###############################################################################
    # Train the neural network
    ###############################################################################
    logging.info(f"Training neural network, iteration {i+1}...")
    train_nn(
      network=p_v_network,
      replay_buffer=replay_buffer,
      epochs=15,
      batch_size=256,
      lr=1e-3,
      weight_decay=1e-4
    )

    ###############################################################################
    # Save the neural network
    ###############################################################################
    torch.save(p_v_network.state_dict(),
               f'strategy/MCTS/models/network_iter_{i + 1}_size_{boarder_size}.pth')
    torch.save(p_v_network.state_dict(),
               f'strategy/MCTS/models/network_latest_size_{boarder_size}.pth')


if __name__ == "__main__":
  main()
