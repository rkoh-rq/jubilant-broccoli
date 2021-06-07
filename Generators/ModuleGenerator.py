import numpy as np
from typing import Tuple

class Module_generator:
  def __init__(self, cell_generator, seed = 200, shape_in_numcells : Tuple[int, int] = (6, 12)):
    '''
    seed: to seed the generator
    shape_cells: size of module to be generated (in terms of number of cells column/row-wise)
    '''
    self.cell_generator = cell_generator
    self.seed = seed
    self.shape_in_numcells = shape_in_numcells
    self.shape_in_px = (shape_in_numcells[0] * cell_generator.shape[0], shape_in_numcells[1] * cell_generator.shape[1], 3)

  def __iter__(self):
    return self

  def __next__(self):
    module = np.empty(self.shape_in_px)
    for r in range(0, self.shape_in_numcells[0]):
      for c in range(0, self.shape_in_numcells[1]):
        module[r * self.cell_generator.shape[0]: (r+1) * self.cell_generator.shape[0], c * self.cell_generator.shape[1]: (c+1) * self.cell_generator.shape[1], :] = next(self.cell_generator)
    return module #, camera_matrix