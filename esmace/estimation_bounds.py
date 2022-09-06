import numpy as np


def hoeffding_bounds(sample_mean, n_points, p=0.025):
    """ Compute the lower and upper hoeffding bounds for the estimated mean arm.sample_mean() with probability p.
    The lower bound satisfies $P[\mu < \hat{\mu} + l_b] \leq p$ where p = $e^{-2t U_t(a)^2} and $l_b = \sqrt(\frac{-log(p)}{2N_t(a)})
    
    
    See https://lilianweng.github.io/posts/2018-01-23-multi-armed-bandit/

    Args:
        arm (Arm): an Arm instance.
        p (float, optional): probability for hoeffding bound. Defaults to 0.025.

    Returns:
        (float, float): lower and upper bounds
    """

    numerator = -np.log(p)
    denominator = 2 * n_points

    bound = np.sqrt(numerator / denominator)
    estimated_mean = sample_mean

    return estimated_mean - bound, estimated_mean + bound
