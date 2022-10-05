import numpy as np
from scipy.stats import norm

np.set_printoptions(suppress=True)


class Tuner(object):
    def __init__(self, one_intercepts, num_samples, sigma):
        self.sigma = sigma
        self.samples, self.normal_likelihoods = self.sample_truncated_gaussian(num_samples)
        self.one_intercepts = one_intercepts
        self.x_intercepts = 1. - 2. / self.one_intercepts
        self.slopes = self.one_intercepts / (1. - self.x_intercepts)
        self.b_values = self.one_intercepts - self.slopes

    def compute_best(self):
        likelihoods = self.slopes[None, :] * self.samples[:, None] + self.b_values[None, :]
        likelihoods = np.clip(likelihoods, a_min=0, a_max=np.inf)
        residuals = abs(likelihoods - self.normal_likelihoods[:, None])
        total_residuals = residuals.sum(axis=0)
        best_idx = np.argmin(total_residuals)
        return self.one_intercepts[best_idx], self.x_intercepts[best_idx], self.slopes[best_idx], self.b_values[best_idx]

    def sample_truncated_gaussian(self, num_samples):
        samples = np.linspace(1-5*self.sigma, 1., num_samples)
        likelihood = 2 * norm.pdf(samples, loc=1., scale=self.sigma)
        return samples, likelihood  # bias toward difficulty is 1
        
if __name__ == '__main__':
    one_intercepts = np.arange(1, 200, 0.01)
    num_samples = 1000
    sigma = 0.005
    tuner = Tuner(one_intercepts, num_samples, sigma)
    one_intercept, x_intercept, slope, b_value = tuner.compute_best()
    print(f'x intercept: {x_intercept}, y intercept: {b_value}, one intercept: {one_intercept}, slope: {slope}')
