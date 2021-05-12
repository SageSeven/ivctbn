import numpy as np


# continuous np.vstack has O(n**2) time flexibility. We need a container.
class IvSampleContainer:
    def __init__(self, col_num):
        self.col_num = col_num
        self.max_len = 1024
        self._container = np.zeros((self.max_len, self.col_num))
        self.cur_row = 0

    def append(self, row):
        self._container[self.cur_row] = row
        self.cur_row += 1
        if self.cur_row == self.max_len:
            new_max_len = self.max_len * 2
            new_container = np.zeros((new_max_len, self.col_num))
            new_container[:self.max_len] = self._container[:self.max_len]
            self.max_len = new_max_len
            self._container = new_container

    def get_values(self):
        return self._container[:self.cur_row]

