import numpy as np


class NoiseInterface:
    def noise(self):
        pass


class GaussianNoise(NoiseInterface):
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def noise(self):
        return np.random.normal(self.mean, np.sqrt(self.variance))


def add_noise