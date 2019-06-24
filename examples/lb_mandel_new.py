from compyle.api import annotate, Elementwise, wrap, get_config, declare
from pyzoltan.hetero.partition_manager import PartitionManager
from pyzoltan.hetero.cell_manager import CellManager
from pyzoltan.hetero.object_exchange import ObjectExchange
from pyzoltan.hetero.utils import make_context, profile, reduce_time
from compyle.cuda import get_context

from pyzoltan.hetero.parallel_manager import only_root, ParallelManager
from pyzoltan.hetero.rcb import dbg_print, root_print
import compyle.array as carr
import numpy as np
import time


@annotate
def mandelbrot_knl(i, qreal, qimag, output, maxiter):
    real = 0.
    imag = 0.
    nreal = 0.
    output[i] = 0
    qr = qreal[i]
    qi = qimag[i]

    for j in range(maxiter):
        nreal = real * real - imag * imag + qr
        imag = 2 * real * imag + qi
        real = nreal

        if real * real + imag * imag > 4.:
            output[i] = j


class MyObjectExchange(ObjectExchange):
    def __init__(self, qreal, qimag, output, backend):
        self.qreal = qreal
        self.qimag = qimag
        self.output = output
        self.backend = backend

    def transfer(self):
        qreal_new = carr.empty(self.plan.nreturn, np.float32,
                           backend=self.backend)
        qimag_new = carr.empty(self.plan.nreturn, np.float32,
                           backend=self.backend)
        output_new = carr.empty(self.plan.nreturn, np.int32,
                           backend=self.backend)

        self.plan.comm_do_post(self.qreal, qreal_new)
        self.plan.comm_do_post(self.qimag, qimag_new)
        self.plan.comm_do_post(self.output, output_new)

        self.plan.comm_do_wait()

        self.qreal = qreal_new
        self.qimag = qimag_new
        self.output = output_new

        return self.qreal, self.qimag


class Solver(object):
    def __init__(self, h, w, lbfreq=10):
        #self.backend = 'cuda'
        self.backend = 'cython'
        #self.ctx = make_context()
        self.func = Elementwise(mandelbrot_knl, backend=self.backend)
        self.lbfreq = lbfreq
        self.h, self.w = h, w
        self.qreal, self.qimag = None, None
        self.output = None
        self.init_arrays(-2.13, 0.77, -1.3, 1.3)
        self.init_partition_manager()

    def init_partition_manager(self):
        self.pm = PartitionManager(2, np.float32, migrate=False, backend=self.backend)
        self.oe = MyObjectExchange(self.qreal, self.qimag, self.output, self.backend)
        cm = CellManager(2, np.float32, 0.1, backend=self.backend)
        self.pm.set_lbfreq(self.lbfreq)
        self.pm.set_object_exchange(self.oe)
        self.pm.set_cell_manager(cm)
        self.pm.setup_load_balancer()

    @only_root
    def init_arrays(self, x1, x2, y1, y2):
        xnum = self.w / (x2 - x1)
        ynum = self.h / (y2 - y1)
        qreal, qimag = np.mgrid[x1:x2:xnum*1j, y1:y2:ynum*1j]
        qreal, qimag = qreal.ravel(), qimag.ravel()
        qreal, qimag = qreal.astype(np.float32), qimag.astype(np.float32)
        output = np.zeros(qreal.size, dtype=np.int32)
        self.qreal, self.qimag = wrap(qreal, qimag, backend=self.backend)
        self.output = wrap(output, backend=self.backend)

    def step(self, qreal, qimag, output, niter):
        self.func(qreal, qimag, output, niter)

    def solve(self, maxiter):
        self.pm.update(self.oe.qreal, self.oe.qimag)
        self.step(self.oe.qreal, self.oe.qimag, self.oe.output, maxiter)
        self.oe.gather()

    @only_root
    def plot(self):
        try:
            import six.moves.tkinter as tk
        except ImportError:
            # Python 3
            import tkinter as tk
        from PIL import Image, ImageTk

        self.mandel = (self.output.reshape((h, w)) /
                           float(self.output.max()) * 255.).astype(np.uint8)

        self.root = tk.Tk()
        self.root.title("Mandelbrot Set")
        self.create_image()
        self.create_label()
        # start event loop
        self.root.mainloop()


#@profile(filename="profile_out")
def run():
    #solver = Solver(16384, 16384)
    solver = Solver(8192, 8192)
    start = time.time()
    solver.solve(65536)
    end = time.time()
    print("Time = %s" % reduce_time(end - start))
    #solver.ctx.pop()


if __name__ == "__main__":
    run()
