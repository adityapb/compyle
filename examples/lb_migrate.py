from compyle.api import annotate, Elementwise, wrap, get_config, declare
from pyzoltan.hetero.partition_manager import PartitionManager
from pyzoltan.hetero.cell_manager import CellManager
from pyzoltan.hetero.object_exchange import ObjectExchange
from compyle.cuda import get_context

from pyzoltan.hetero.parallel_manager import only_root, ParallelManager
from pyzoltan.hetero.rcb import dbg_print, root_print
from pyzoltan.hetero.utils import make_context, profile, reduce_time
import compyle.array as carr
import numpy as np
import time


class MyObjectExchange(ObjectExchange):
    def __init__(self, x, y, vx, vy, backend):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

        #self.x_alt = carr.empty(1, np.float32, backend=backend)
        #self.y_alt = carr.empty(1, np.float32, backend=backend)
        #self.vx_alt = carr.empty(1, np.float32, backend=backend)
        #self.vy_alt = carr.empty(1, np.float32, backend=backend)
        #self.colors_alt = carr.empty(1, np.int32, backend=backend)

        self.backend = backend

    def transfer(self):
        #dbg_print("%s %s" % (self.plan.start_from, self.plan.lengths_from))
        x_alt = carr.empty(self.plan.nreturn, np.float32,
                                backend=self.backend)
        y_alt = carr.empty(self.plan.nreturn, np.float32,
                                backend=self.backend)
        vx_alt = carr.empty(self.plan.nreturn, np.float32,
                                 backend=self.backend)
        vy_alt = carr.empty(self.plan.nreturn, np.float32,
                                 backend=self.backend)

        self.plan.comm_do_post(self.x, x_alt)
        self.plan.comm_do_post(self.y, y_alt)
        self.plan.comm_do_post(self.vx, vx_alt)
        self.plan.comm_do_post(self.vy, vy_alt)

        self.plan.comm_do_wait()

        #self.x, self.x_alt = self.x_alt, self.x
        #self.y, self.y_alt = self.y_alt, self.y
        #self.vx, self.vx_alt = self.vx_alt, self.vx
        #self.vy, self.vy_alt = self.vy_alt, self.vy
        #self.colors, self.colors_alt = self.colors_alt, self.colors

        self.x = x_alt
        self.y = y_alt
        self.vx = vx_alt
        self.vy = vy_alt

        return self.x, self.y


@annotate
def step_euler(i, x, y, vx, vy, dt, size):
    x[i] += vx[i] * dt
    y[i] += vy[i] * dt

    vx[i] = -y[i]
    vy[i] = x[i]

    count = 0

    for j in range(size):
        dij2 = x[i] * x[i] + y[i] * y[i]

        if dij2 < 0.1:
            count += 1


class Solver(object):
    def __init__(self, n, dt, lbfreq=20, animate=False):
        self.backend = 'cython'
        #self.ctx = make_context()
        self.step_func = Elementwise(step_euler, backend=self.backend)
        self.n = n
        self.dt = dt
        self.lbfreq = lbfreq
        self.x, self.y = None, None
        self.vx, self.vy = None, None
        self.colors = None
        self.animate = animate
        self.init_arrays()
        self.init_partition_manager()

    def init_partition_manager(self):
        self.pm = PartitionManager(2, np.float32, backend=self.backend)
        self.oe = MyObjectExchange(self.x, self.y, self.vx, self.vy,
                                   self.backend)
        cm = CellManager(2, np.float32, 0.01, padding=0.1,
                         backend=self.backend)
        self.pm.set_lbfreq(self.lbfreq)
        self.pm.set_object_exchange(self.oe)
        self.pm.set_cell_manager(cm)
        self.pm.setup_load_balancer()

    @only_root
    def init_arrays(self):
        x, y = np.mgrid[-1:1:self.n*1j, -1:1:self.n*1j]
        x, y = x.ravel(), y.ravel()

        x = x.astype(np.float32)
        y = y.astype(np.float32)

        vy = x.copy()
        vx = -1 * y.copy()
        self.x, self.y, self.vx, self.vy = wrap(x, y, vx, vy,
                                                backend=self.backend)

    def step(self, x, y, vx, vy):
        self.step_func(x, y, vx, vy, self.dt, 9000000)

    def set_colors(self, colors):
        colors[:] = self.pm.rank

    @profile(filename="profile_cpu")
    def solve(self, niter):
        for i in range(niter):
            self.pm.update(self.oe.x, self.oe.y, migrate=True)
            self.step(self.oe.x, self.oe.y, self.oe.vx, self.oe.vy)
            if self.animate:
                self.oe.gather()
                self.animate_plot()
        #dbg_print("%s %s" % (self.oe.x, self.oe.y))
        self.oe.gather()
        #dbg_print(self.oe.colors)

    @only_root
    def animate_plot(self):
        pass

    @only_root
    def plot(self):
        import matplotlib.pyplot as plt
        x, y = self.oe.x, self.oe.y
        cmap = np.array(['r', 'b', 'g'])
        plt.scatter(x, y, c=cmap[self.oe.colors])
        plt.show()


def run():
    solver = Solver(3000, 0.01)
    start = time.time()
    solver.solve(40)
    total = time.time() - start
    print(reduce_time(total))
    #solver.plot()
    #solver.ctx.pop()


if __name__ == "__main__":
    run()
