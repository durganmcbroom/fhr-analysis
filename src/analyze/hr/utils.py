import numpy as np




def _gaussian_smooth(x: np.ndarray, sigma_samples: float) -> np.ndarray:
    sigma_samples = max(1.0, float(sigma_samples))
    radius = int(3 * sigma_samples)
    grid = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (grid / sigma_samples) ** 2)
    kernel /= kernel.sum()
    return np.convolve(x, kernel, mode='same')


def _impulse_train(times: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    impulse = np.zeros_like(t_grid, dtype=float)
    if len(times) == 0:
        return impulse
    dt = t_grid[1] - t_grid[0]
    idx = np.round((times - t_grid[0]) / dt).astype(int)
    idx = idx[(idx >= 0) & (idx < len(t_grid))]
    impulse[idx] = 1.0
    return impulse
