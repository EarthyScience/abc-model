import numpy as np


def esat(T):
    return 0.611e3 * np.exp(17.2694 * (T - 273.16) / (T - 35.86))


def qsat(T, p):
    return 0.622 * esat(T) / p
