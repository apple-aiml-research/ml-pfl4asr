#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import numpy as np


class CounterValue:
    """
    Average value counter
    """

    def __init__(self):
        self.total_value = 0
        self.total_num = 0

    def add(self, val, n=1):
        self.total_value += val
        self.total_num += n

    def get_average(self):
        return self.total_value / (self.total_num + 1e-6)


class CounterArray:
    """
    Average and std counter for arrays
    """

    def __init__(self):
        self.data = None

    def add(self, val):
        if self.data is None:
            self.data = [val]
        else:
            self.data.append(val)

    def get_std(self):
        if self.data is None:
            return 0
        return np.std(self.data)

    def get_average(self):
        if self.data is None:
            return 0
        return np.mean(self.data)

    def get_num(self):
        return len(self.data)

    def reset(self):
        self.data = None
