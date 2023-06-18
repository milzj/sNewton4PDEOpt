import numpy as np
from scipy.stats import truncnorm

class Sampler(object):
    """Sampler for a real-valued random diffusion coefficient.
    """

    def __init__(self):
        """
        """
        self.seed = 1

        # Define scalar random diffusion coefficient

        l = .5
        u = 3.5
        mu = 2.
        std = .25

        a, b = (l - mu) / std, (u - mu) / std

        self.a = a
        self.b = b
        self.l = l
        self.u = u
        self.mu = mu
        self.std = std

        self.means = [0.2627515620144445,  0.5082109850109712, 0.0]

    def bump_seed(self):
        self.seed += 1

    def sample(self, number_sample):
        """Generates a sample

        Parameters:
        -----------
            number_sample : int
                number of samples used 

        Returns: 
        --------
            sample : ndarray			
        """

        self.bump_seed()
        np.random.seed(self.seed)

        Z = truncnorm.rvs(self.a, self.b, \
            loc = self.mu, scale = self.std, size=number_sample)

        self.bump_seed()
        np.random.seed(self.seed)
        W = -1.0+2.0*np.random.rand(number_sample)

        Z_2_inv = np.mean(1.0/Z**2)

        return [Z_2_inv, np.mean(1.0/Z), np.mean(W/Z**2)/Z_2_inv]


if __name__ == "__main__":

    sampler = Sampler()

    N = 1000
    s = sampler.sample(N)
    means = np.array(sampler.means)
    rel_err = np.linalg.norm(s-means)/np.linalg.norm(means)
    print(rel_err)
