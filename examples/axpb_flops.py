from compyle.api import Elementwise, annotate, wrap, get_config
import numpy as np
from numpy import sin
import time
from pypapi import papi_high
from pypapi import events as papi_events


@annotate(int='i, size', doublep='x, y, a, b')
def axpb(i, x, y, a, b, size):
    result = declare('double')
    result = 0.
    for j in range(size):
        result += a[i]*x[j] + b[i]
    y[i] = result


def setup(backend, openmp=False):
    get_config().use_openmp = openmp
    e = Elementwise(axpb, backend=backend)
    return e


def data(n, backend):
    x = np.linspace(0, 1, n)
    y = np.zeros_like(x)
    a = x*x
    b = np.sqrt(x + 1)
    return wrap(x, y, a, b, backend=backend)


if __name__ == '__main__':
    backend = 'cython'
    knl = setup(backend)
    x, y, a, b = data(1000, backend)

    papi_high.start_counters([
        papi_events.PAPI_DP_OPS
    ])

    results = papi_high.read_counters()
    knl(x, y, a, b, x.length)
    results = papi_high.stop_counters()
    print(results)
