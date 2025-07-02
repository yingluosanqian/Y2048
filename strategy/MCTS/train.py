
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
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

device = "cuda" if torch.cuda.is_available() else "cpu"


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
  strategy, network, buffer_size = args
  local_buffer = ReplayBuffer(max_size=buffer_size)
  while len(local_buffer) < buffer_size:
    strategy.collect_trajectory(local_buffer, network=network, gui=False)
  return local_buffer


def collect_train_data(network, replay_buffer: ReplayBuffer, num_workers=10):
  with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
    # Each worker gets its own Strategy and collects samples
    sub_buffer_size = replay_buffer.max_size // num_workers
    args = [(Strategy(), copy.deepcopy(network), sub_buffer_size) for _ in range(num_workers)]
    results = pool.map(_collect_trajectory_worker, args)
    # Flatten and add to replay_buffer

    for worker_samples in results:
      replay_buffer.extend(worker_samples)


def get_dataloader_from_replaybuffer(replay_buffer, batch_size=128, shuffle=True):
  dataset = ReplayBufferDataset(replay_buffer)
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_nn(
  network: PolicyValueNet,
  replay_buffer: ReplayBuffer,
  epochs=20,
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
      # loss_value = value_loss(value_pred, value_target)
      total_loss = loss_policy
      # total_loss = loss_policy + loss_value

      total_loss.backward()
      optimizer.step()

      total_policy_loss += loss_policy.item()
      # total_value_loss += loss_value.item()
    avg_policy_loss = total_policy_loss / len(dataloader)
    # avg_value_loss = total_value_loss / len(dataloader)
    logging.info(f"Epoch {epoch+1}: Policy Loss = {avg_policy_loss:.4f}")


def eval_models(network: PolicyValueNet, replay_buffer: ReplayBuffer):
  strategy = Strategy()
  scores = []
  for i in tqdm(range(20), desc="Evaluating models"):
    score = strategy.collect_trajectory(
      replay_buffer, network=network, gui=False, collect=False)
    scores.append(score)
  avg_score = np.mean(scores)
  return avg_score


def main():
  logging.info("====================================================")
  logging.info("Starting training process...")
  logging.info("====================================================")
  network = PolicyValueNet(3, 3, num_res_blocks=2).to(device)  # neural network
  # best_avg_score = eval_models(network, None)
  best_avg_score = 0.0  # Initialize best average score
  logging.info(f"Initial average score: {best_avg_score:.2f}")

  network_path = Path('strategy/MCTS/models/network_latest.pth')
  if network_path.exists():
    logging.info("Loading existing network weights...")
    torch.load('strategy/MCTS/models/network_latest.pth',
               network.state_dict(), weights_only=True)
  else:
    logging.info("No existing network weights found, starting from scratch.")

  for i in range(100):
    replay_buffer = ReplayBuffer()

    logging.info(f"Collecting training data, iteration {i+1}...")
    collect_train_data(network=network, replay_buffer=replay_buffer)

    logging.info(f"Training neural network, iteration {i+1}...")
    train_nn(
      network=network,
      replay_buffer=replay_buffer,
      epochs=5,
      batch_size=128,
      lr=1e-3,
      weight_decay=1e-4
    )

    if (i + 1) % 10 == 0:
      logging.info(f"Evaluating model after {i+1} iterations...")
      avg_score = eval_models(network, None)
      logging.info(f"Average score after {i+1} iterations: {avg_score:.2f}")
      if avg_score > best_avg_score * 1.05:  # 5% improvement threshold
        # Save the model if it improves by 5% or more
        logging.info(
          f"Improvement detected! Saving model with score: {avg_score:.2f}. (Old: {best_avg_score:.2f})")
        best_avg_score = avg_score
        torch.save(network.state_dict(),
                   'strategy/MCTS/models/network_latest.pth')
      else:
        logging.info(
          f"No improvement, best score remains: {best_avg_score:.2f}")


if __name__ == "__main__":
  main()
