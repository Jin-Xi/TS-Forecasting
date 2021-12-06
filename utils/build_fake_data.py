import numpy as np
import matplotlib.pyplot as plt

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0.1):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def get_series(data_num: int,
                base = 10,
                amplitude = 40,
                slope = 0.05,
                noise_level = 5,seed=42):

    time = np.arange(data_num, dtype="float32")
    series = trend(time, 0.1)
    # Create the series
    series = base + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
    # Update with noise
    series += noise(time, noise_level, seed=seed)

    return series

if __name__ == "__main__":
    se = get_series(4000)
    plt.plot(se)
    plt.show()
    # time = np.arange(4000, dtype="float32")
    # time * 0.1
