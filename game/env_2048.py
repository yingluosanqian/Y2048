import numpy as np
import random

from types import SimpleNamespace
from gym import spaces
from const import const_boarder_size as boarder_size


class Env2048:
  def __init__(self, size=boarder_size, seed=None):
    self.size = size
    if seed is not None:
      self.rng = random.Random(seed)
    else:
      self.rng = random.Random()
    self.reset()

  ###########################################
  # Method for the environment
  ###########################################
  def step(self, action) -> tuple[dict, float, bool, dict]:
    moved, score_gain = self._merge_all(action)

    if moved:
      self._add_new_tile()

    return self.observation_space.matrix, float(score_gain), self._is_game_over(), SimpleNamespace(**{"moved": moved})

  def reset(self):
    self.matrix = [[0] * self.size for _ in range(self.size)]
    self.score = 0
    self._add_new_tile()
    self._add_new_tile()
    return self.observation_space.matrix

  def render(self):
    print("Env 2048 State:")
    for row in self.matrix:
      print(" ".join(f"{num:>5}" for num in row))

  def close(self):
    pass

  ###########################################
  # Attributes for the environment
  ###########################################

  @property
  def action_space(self):
    return spaces.Discrete(4)

  @property
  def observation_space(self):
    return SimpleNamespace(**{
      'matrix': self.matrix,
    })

  ###########################################
  # Help Function
  ###########################################

  def _add_new_tile(self, pos_x=None, pos_y=None, value=None):

    empty_tiles = [(i, j) for i in range(self.size)
                   for j in range(self.size) if self.matrix[i][j] == 0]
    if empty_tiles:
      i, j = self.rng.choice(empty_tiles)
      i = i if pos_x is None else pos_x
      j = j if pos_y is None else pos_y

      self.matrix[i][j] = 2 if self.rng.random() < 0.9 else 4
      self.matrix[i][j] = value if value is not None else self.matrix[i][j]

  def _merge_all(self, action) -> tuple[bool, int]:
    moved = False
    score_gain = 0

    if action == 'Left' or action == 0:
      for i in range(self.size):
        merged, local_score = self._merge(self.matrix[i])
        if merged != self.matrix[i]:
          moved = True
        self.matrix[i] = merged
        score_gain += local_score
    elif action == 'Right' or action == 1:
      for i in range(self.size):
        reversed_row = self.matrix[i][::-1]
        merged, local_score = self._merge(reversed_row)
        merged = merged[::-1]
        if merged != self.matrix[i]:
          moved = True
        self.matrix[i] = merged
        score_gain += local_score
    elif action == 'Up' or action == 2:
      for j in range(self.size):
        col = [self.matrix[i][j] for i in range(self.size)]
        merged, local_score = self._merge(col)
        for i in range(self.size):
          if self.matrix[i][j] != merged[i]:
            moved = True
          self.matrix[i][j] = merged[i]
        score_gain += local_score
    elif action == 'Down' or action == 3:
      for j in range(self.size):
        col = [self.matrix[i][j] for i in range(self.size)][::-1]
        merged, local_score = self._merge(col)
        merged = merged[::-1]
        for i in range(self.size):
          if self.matrix[i][j] != merged[i]:
            moved = True
          self.matrix[i][j] = merged[i]
        score_gain += local_score
    return moved, score_gain

  def _merge(self, row: int) -> int:
    merged = []
    local_score = 0
    last = -1
    for num in row + [-2]:
      if num == 0:
        continue
      if num == last:
        merged.append(num * 2)
        local_score += num * 2
        last = -1
      else:
        if last != -1:
          merged.append(last)
        last = num

    merged += [0] * (self.size - len(merged))
    return merged, local_score

  def _is_game_over(self):
    for i in range(self.size):
      for j in range(self.size):
        if self.matrix[i][j] == 0:
          return False
        if i < self.size - 1 and self.matrix[i][j] == self.matrix[i + 1][j]:
          return False
        if j < self.size - 1 and self.matrix[i][j] == self.matrix[i][j + 1]:
          return False
    return True
  
  def _eval_value(self):
    score = 0.0
    max_level = 17
    max_in_matrix = max(max(row) for row in self.matrix)
    score += np.log2(max_in_matrix + 1) / max_level
    sum_of_tiles = sum(sum(row) for row in self.matrix) - max_in_matrix
    score += min(1, sum_of_tiles / max_in_matrix) * (1 / 17)
    return score
  
  @classmethod
  def rotate(cls, matrix, policy, rotate_times):
    for _ in range(rotate_times % 4):
      matrix = [list(row) for row in zip(*matrix[::-1])]
      policy = [policy[3], policy[2], policy[0], policy[1]]
    return matrix, policy
