import torch
import torch.nn as nn
import torch.nn.functional as F

from const import const_device as device


class ConvBlock(nn.Module):
  def __init__(self, in_channels=18, out_channels=256, kernel_size=3):
    super(ConvBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels,
                           kernel_size=kernel_size, stride=1, padding=1, dtype=torch.float32)
    self.bn = nn.BatchNorm2d(out_channels, dtype=torch.float32)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn(x)
    x = F.relu(x)
    return x


class ResidualBlock(nn.Module):
  def __init__(self, num_channels=256):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(num_channels, num_channels,
                           kernel_size=3, stride=1, padding=1, dtype=torch.float32)
    self.bn1 = nn.BatchNorm2d(num_channels, dtype=torch.float32)
    self.conv2 = nn.Conv2d(num_channels, num_channels,
                           kernel_size=3, stride=1, padding=1, dtype=torch.float32)
    self.bn2 = nn.BatchNorm2d(num_channels, dtype=torch.float32)

  def forward(self, x):
    residual = x
    x = self.conv1(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x += residual
    x = F.relu(x)
    return x


class PolicyValueHeads(nn.Module):
  def __init__(self, grid_n, grid_m):
    super(PolicyValueHeads, self).__init__()
    self.board_size = grid_n * grid_m

    # Policy Head
    self.policy_conv = nn.Conv2d(256, 2, kernel_size=1, stride=1, dtype=torch.float32)
    self.policy_bn = nn.BatchNorm2d(2, dtype=torch.float32)
    self.policy_fc = nn.Linear(2 * self.board_size, 4, dtype=torch.float32)  # 4-direction

    # Value Head
    self.value_conv = nn.Conv2d(256, 1, kernel_size=1, stride=1, dtype=torch.float32)
    self.value_bn = nn.BatchNorm2d(1, dtype=torch.float32)
    self.value_fc1 = nn.Linear(self.board_size, 256, dtype=torch.float32)  # with flatten input
    self.value_fc2 = nn.Linear(256, 1, dtype=torch.float32)

  def forward(self, x):
    # Shape of x: [batch, 256, grid_n, grid_m]

    # --- Policy Head ---
    p = F.relu(self.policy_bn(self.policy_conv(x)))  # [B, 2, grid_n, grid_m]
    p = p.view(-1, 2 * self.board_size)  # flatten
    policy_logits = self.policy_fc(p)  # [B, 4]

    # --- Value Head ---
    v = F.relu(self.value_bn(self.value_conv(x)))  # [B, 1, grid_n, grid_m]
    v = v.view(-1, self.board_size)  # flatten
    v = F.relu(self.value_fc1(v))
    v = self.value_fc2(v)
    value = torch.tanh(v)  # [-1, 1]

    return policy_logits, value


class PolicyValueNet(nn.Module):
  def __init__(self, grid_n, grid_m, num_res_blocks=19):
    super().__init__()
    self.conv_block = ConvBlock(in_channels=18)
    self.res_tower = nn.Sequential(
      *[ResidualBlock() for _ in range(num_res_blocks)])
    self.policy_value_heads = PolicyValueHeads(grid_n, grid_m)

  def forward(self, x):
    # Shape of x: [batch, 18, grid_n, grid_m]
    x = self.conv_block(x)  # [B, 256, grid_n, grid_m]
    x = self.res_tower(x)  # [B, 256, grid_n, grid_m]
    return self.policy_value_heads(x)

  def take_action(self, x):
    x = x.unsqueeze(0)  # Add batch dimension if not present
    policy_logits, _ = self.forward(x)
    return policy_logits.argmax().item()


# Test
if __name__ == "__main__":
  grid_n, grid_m = 3, 3  # 4x4 grid for 2048
  net = PolicyValueNet(grid_n, grid_m, num_res_blocks=2).to(device)
  dummy_input = torch.randn(18, grid_n, grid_m, dtype=torch.float32, device=device)  # mock
  
  import time
  start_time = time.time()
  net.eval()  # Set to evaluation mode
  for _ in range(10):
    _ = net.take_action(dummy_input)
  end_time = time.time()
  print(f"Forward pass time: {end_time - start_time:.6f} seconds")

  # print(f"Shape of policy-head: {policy.shape}")  # Should be [1, 4]
  # # Should be in [-1, 1]
  # print(f"Shape of value-head: [{value.min():.3f}, {value.max():.3f}]")
