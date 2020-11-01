import tensorflow as tf
import numpy as np
import os
import time
import shutil
from SplittingModel import simulate


def phi(y):
    return 1. / (2. + 2. / 5. * tf.reduce_sum(y ** 2, axis=1, keepdims=True))


def f(x, y, z):
    return tf.sin(y)


def sde(_d, n):
    x = [tf.random_normal([batch_size, _d, 1], stddev=np.sqrt(2. * n * T / N)),
         tf.random_normal([batch_size, _d, 1], stddev=np.sqrt(2. * T / N))]
    return tf.cumsum(tf.concat(x, axis=2), axis=2)


N, T = 20, 0.3
batch_size = 256
train_steps = 1000
lr_boundaries = [250, 500, 750]
lr_values = [0.1, 0.01, 0.001, 0.0001]

_file = open('SineGordon.csv', 'w')
_file.write('d, T, N, run, value, time\n')

for d in [10, 50, 100, 200, 300, 500, 1000, 5000, 10000]:

    neurons = [d + 50, d + 50, 1]

    for run in range(10):

        path = '/tmp/sinegordon'
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        t_0 = time.time()
        v_n = simulate(T, N, d, sde, phi, f, neurons, train_steps, batch_size,
                       lr_boundaries, lr_values, path)
        t_1 = time.time()

        _file.write('%i, %f, %i, %i, %f, %f\n'
                    % (d, T, N, run, v_n, t_1 - t_0))

_file.close()
