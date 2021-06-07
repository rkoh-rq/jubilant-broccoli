import numpy as np
from typing import Tuple

class Fake_cell_generator:
  '''
  Just here for now, to be replaced by generator of real cells
  '''
  def __init__(self, seed = 200, shape : Tuple[int, int] = (50, 50), line_width=1):
    '''
    seed: to seed the generator
    shape: size of cell to be generated (in pixels)
    '''
    self.seed = seed
    self.shape = shape
    self.line_width = line_width
    np.random.seed(self.seed)

  def __iter__(self):
    np.random.seed(self.seed)
    return self

  def __next__(self):
    # r, g, b = np.random.randint(0, 255, size=3)
    # fake_cell = np.array([r, g, b], dtype=np.float32) * np.ones((self.shape[0], self.shape[1], 3))
    r = np.random.normal(150, 50)
    r = max(r, 0)
    r = min(r, 255)
    fake_cell = np.array([r, r, r], dtype=np.float32) * np.ones((self.shape[0], self.shape[1], 3))
    # Adding lines on the
    # top
    fake_cell[:self.line_width, :, :] = np.ones((self.line_width, self.shape[1], 3))
    # bottom
    fake_cell[self.shape[0] - self.line_width:, :, :] = np.ones((self.line_width, self.shape[1], 3))
    # left
    fake_cell[:, :self.line_width, :] = np.ones((self.shape[1], self.line_width, 3))
    # right
    fake_cell[:, self.shape[1] - self.line_width:, :] = np.ones((self.shape[1], self.line_width, 3))
    return fake_cell