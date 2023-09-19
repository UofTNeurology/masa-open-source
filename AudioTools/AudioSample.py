import numpy as np
import warnings

class AudioSample:

    def __init__(self, data, sampling_rate, min_power=None, name="Audio"):
        self.data = data
        self.sampling_rate = sampling_rate
        self.num_samples = len(self.data)
        self.duration = self.num_samples / self.sampling_rate
        self.num_channels = 2 if len(self.data.shape) == 2 else 1
        self.name = name
        self.min_power = min_power
        self.power = None

    def calculate_power(self):
        self.power = np.sum(np.abs(self.data) ** 2)
        return self.power

    def check_power(self, min_power=None):
        if min_power is not None:
            self.min_power = min_power

        if self.power < self.min_power:
            warnings.warn(
                "POWER WARNING: Audio sample has a power of " + str(self.power) + " the minimum power is " +
                str(min_power))

        return self.power


