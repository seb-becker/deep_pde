import tensorflow as tf
import numpy as np
import os
import time
import shutil
from SplittingModel import simulate


def phi(y):
    return tf.reduce_min(y, axis=1, keepdims=True)


def f(x, y, z):
    return -(1. - delta) * tf.minimum(tf.maximum((y - v_h) * (gamma_h - gamma_l) / (v_h - v_l) + gamma_h, gamma_l), gamma_h) * y - R * y


def sde(_d, n):
    y = [tf.ones((batch_size, _d)) * 50.]
    for _n in range(n + 1):
        y.append(y[-1] * (1. + mu_bar * T / N + sigma_bar * tf.random_normal((batch_size, _d), stddev=np.sqrt(T / N))))
    return tf.stack(y[n:n + 2], axis=2)


def sde_loop(_d, n):
    xi = tf.ones((batch_size, _d)) * 50.

    def loop(_n, _x0, _x1):
        _x0 = _x1
        _x1 = _x1 * (1. + mu_bar * T / N + sigma_bar * tf.random_normal((batch_size, _d), stddev=np.sqrt(T / N)))
        return _n + 1, _x0, _x1

    _, x0, x1 = tf.while_loop(lambda _n, _x0, _x1: _n <= n, loop, (tf.constant(0), xi, xi))

    return tf.stack([x0, x1], axis=2)


N, T = 96, 1. / 3.
delta, R = 2. / 3., 0.02
mu_bar, sigma_bar = 0.02, 0.2
v_h, v_l = 50., 70.
gamma_h, gamma_l = 0.2, 0.02
lr_values = [0.1, 0.01, 0.001]
_file = open('nonlinear_BS.csv', 'w')
_file.write('d, T, N, run, value, time\n')

for d in [10, 50, 100, 200, 300, 500, 1000, 5000, 10000]:

    neurons = [d + 10, d + 10, 1] if d > 100 else [d + 50, d + 50, 1]
    batch_size = 256 if d > 100 else 4096
    train_steps = 2000 if d > 100 else 3000
    lr_boundaries = [1500, 1750] if d > 100 else [2500, 2750]

    for run in range(1):

        path = '/tmp/bs'
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        t_0 = time.time()
        v_n = simulate(T, N, d, sde_loop if d > 100 else sde, phi, f, neurons, train_steps, batch_size, lr_boundaries,
                       lr_values, path)
        t_1 = time.time()

        _file.write('%i, %f, %i, %i, %f, %f\n'
                    % (d, T, N, run, v_n, t_1 - t_0))
        _file.flush()

_file.close()
