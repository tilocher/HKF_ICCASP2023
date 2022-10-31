import os

import torch
import numpy as np
from PriorModels.BasePrior import BasePrior
from scipy.optimize import least_squares
from SystemModels.ExtendedSysmodel import ExtendedSystemModel

pi = np.pi
_theta = np.array([-pi / 3, -pi / 12, 0, pi / 12, pi / 2]) + pi
_a = np.array([1.2, -5.0, 30, -7.5, 0.75])
_b = np.array([0.25, 0.1, 0.1, 0.1, 0.4])


class PdePrior(BasePrior):

    def __init__(self, channels, fs, expected_period=1):
        super(PdePrior, self).__init__()
        self.channels = channels
        self.fs = fs
        self.expected_period = expected_period

        self.__w = 2 * pi * expected_period
        self.__alpha = _a * _b ** 2 / self.w
        self.__b = _b
        self.__theta = _theta

    @property
    def a(self):
        return self.__alpha * self.w / self.b ** 2

    @property
    def alpha(self):
        return self.__alpha

    @alpha.setter
    def alpha(self, val):
        self.__alpha = val

    @property
    def b(self):
        return self.__b

    @b.setter
    def b(self, val):
        self.__b = val

    @property
    def theta(self):
        return self.__theta

    @theta.setter
    def theta(self, val):
        self.__theta = val

    @property
    def w(self):
        return self.__w

    @w.setter
    def w(self, val):
        self.__w = val

    def run(self, T: int, alphas: np.array, b: np.array, thetas: np.array, w: np.array, z_0: float)\
            -> np.array:
        """
        Run simulation of a heartbeat given the parameters
        :param T: Time horizon
        :param alphas: Peak amplitudes
        :param b: Peak standard deviations
        :param thetas: Peak angle offsets from R-peak
        :param w: Angular velocity of the heartrate
        :param z_0: DC-offset
        :return:
        """
        # Read out input arrays
        self.alpha = alphas
        self.b = b
        self.theta = thetas
        self.w = w

        # Allocate buffer
        simulation = torch.empty(T, 2)

        # Initialize angle and DC-offset
        theta_t = 0
        z_t = z_0

        # Start discrete-time simulation
        for t in range(T):

            # Angle offsets for each wave
            theta_t1 = (theta_t + self.w / self.fs) % (2 * pi)
            d_theta = theta_t - self.theta
            z_t1 = - np.sum(1 / self.fs * self.a * d_theta * np.exp(-d_theta ** 2 / (2 * self.b ** 2)))
            z_t1 = z_t + z_t1

            simulation[t, 0] = theta_t1
            simulation[t, 1] = z_t1

            z_t = z_t1
            theta_t = theta_t1

        return simulation

    @torch.no_grad()
    def fit(self, data: torch.Tensor):

        batch_size, T, channels = data.shape
        self.time_steps = T

        if 'optimized_vector.npy' in os.listdir('tmp'):
            optimization_vector = np.load('tmp/optimized_vector.npy')
            self.alpha = optimization_vector[:5]
            self.b = optimization_vector[5:10]
            self.theta = optimization_vector[10:15]
            self.w = optimization_vector[15]
            self.z_0 = optimization_vector[16]

            return optimization_vector

        else:


            def optimization_loss(parameter_vector: np.array, observations: np.array, model: PdePrior):
                alphas = parameter_vector[:5]
                b = parameter_vector[5:10]
                thetas = parameter_vector[10:15]
                w = parameter_vector[15]
                z_0 = parameter_vector[16]

                simulation = model.run(T, alphas=alphas, b=b, thetas=thetas, w=w, z_0=z_0)

                loss = ((simulation[:, 1] - observations) ** 2).mean()

                return loss

            mean_ecg = data.mean(0).detach().numpy()

            init_vector = np.zeros(17)
            init_vector[:5] = self.alpha
            init_vector[5:10] = self.b
            init_vector[10:15] = self.theta
            init_vector[15] = self.w
            init_vector[16] = 0

            optimization_vector = least_squares(optimization_loss, init_vector,
                                                args=(mean_ecg[:, 0], self), method='dogbox').x

            self.alpha = optimization_vector[:5]
            self.b = optimization_vector[5:10]
            self.theta = optimization_vector[10:15]
            self.w = optimization_vector[15]
            self.z_0 = optimization_vector[16]

            np.save('tmp/optimized_vector.npy', optimization_vector)

            return optimization_vector

    def f(self, x: torch.Tensor, t: int):

        theta_t = (t * self.w / self.fs) % (2 * pi)

        d_theta = theta_t - self.theta

        z_t1 = - np.sum(1 / self.fs * self.a * d_theta * np.exp(-d_theta ** 2 / (2 * self.b ** 2)))
        x = x + z_t1

        return x.reshape(1,1).float()

    def h(self, x: torch.Tensor, t:int):
        return x.reshape(1,1).float()

    def df(self, x:torch.Tensor, t:int):
        return torch.eye(1)
        # theta_t = (t * self.w / self.fs) % (2 * pi)
        #
        # d_theta = theta_t - self.theta
        # df = - np.sum(self.fs * self.a *( 1- d_theta**2/self.b**2) * np.exp(-d_theta**2/(2*self.b**2)))
        # df = max(1e-6, df)
        # return torch.tensor(df).reshape(1,1).float()

    def dh(self, x, t):
        return torch.eye(1).float()

    def get_sys_model(self):

        sys_model = ExtendedSystemModel(self.f, 0, self.h, 0, self.time_steps, 1, 1)
        sys_model.set_f_jacobian(self.df)
        sys_model.set_h_jacobian(self.dh)

        return sys_model

if __name__ == '__main__':
    prior = PdePrior(2, 360, 1)

    from matplotlib import pyplot as plt

    from Dataloaders.MIT_BIH_DataLoader import MIT_BIH_DataLoader
    from utils.GetSubset import get_subset
    from torch.utils.data.dataloader import DataLoader

    dataloader = MIT_BIH_DataLoader(360, [1], 0)
    prior_loader, test_loader = get_subset(dataloader, 10)

    prior_set_length = len(prior_loader)

    observations, _ = next(iter(DataLoader(prior_loader, batch_size=prior_set_length)))

    sim = prior.fit(observations)
    plt.plot(observations.mean(0)[:, 0])
    plt.plot(sim[:, 1])
    plt.show()
