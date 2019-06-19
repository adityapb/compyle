from compyle.api import annotate, Elementwise, wrap, get_config, declare
from pyzoltan.hetero.partition_manager import PartitionManager
from pyzoltan.hetero.cell_manager import CellManager
from pyzoltan.hetero.object_exchange import ObjectExchange
from pyzoltan.hetero.utils import make_context
from compyle.cuda import get_context

from pyzoltan.hetero.parallel_manager import only_root, ParallelManager
from pyzoltan.hetero.rcb import dbg_print, root_print
from pyzoltan.hetero.profile import profile
import compyle.array as carr
import numpy as np


class MyObjectExchange(ObjectExchange):
    def __init__(self, x, y, vx, vy, colors, backend):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.colors = colors

        self.x_alt = carr.empty(1, np.float32, backend=backend)
        self.y_alt = carr.empty(1, np.float32, backend=backend)
        self.vx_alt = carr.empty(1, np.float32, backend=backend)
        self.vy_alt = carr.empty(1, np.float32, backend=backend)
        self.colors_alt = carr.empty(1, np.int32, backend=backend)

        self.backend = backend

    def transfer(self):
        self.x_alt = carr.empty(self.plan.nreturn, np.float32,
                                backend=self.backend)
        self.y_alt = carr.empty(self.plan.nreturn, np.float32,
                                backend=self.backend)
        self.vx_alt = carr.empty(self.plan.nreturn, np.float32,
                                 backend=self.backend)
        self.vy_alt = carr.empty(self.plan.nreturn, np.float32,
                                 backend=self.backend)
        self.colors_alt = carr.empty(self.plan.nreturn, np.int32,
                                     backend=self.backend)

        self.plan.comm_do_post(self.x, self.x_alt)
        self.plan.comm_do_post(self.y, self.y_alt)
        self.plan.comm_do_post(self.vx, self.vx_alt)
        self.plan.comm_do_post(self.vy, self.vy_alt)
        self.plan.comm_do_post(self.colors, self.colors_alt)

        self.plan.comm_do_wait()

        #self.x, self.x_alt = self.x_alt, self.x
        #self.y, self.y_alt = self.y_alt, self.y
        #self.vx, self.vx_alt = self.vx_alt, self.vx
        #self.vy, self.vy_alt = self.vy_alt, self.vy
        #self.colors, self.colors_alt = self.colors_alt, self.colors

        self.x = self.x_alt
        self.y = self.y_alt
        self.vx = self.vx_alt
        self.vy = self.vy_alt
        self.colors = self.colors_alt

        return self.x, self.y


@annotate
def step_euler(i, x, y, vx, vy, dt):
    x[i] += vx[i] * dt
    y[i] += vy[i] * dt

    vx[i] = -y[i]
    vy[i] = x[i]


class Solver(object):
    def __init__(self, n, dt, lbfreq=1000, animate=False):
        self.backend = 'cuda'
        self.ctx = make_context()
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
                                   self.colors, self.backend)
        cm = CellManager(2, np.float32, 0.05, padding=1., backend=self.backend)
        self.pm.set_lbfreq(self.lbfreq)
        self.pm.set_object_exchange(self.oe)
        self.pm.set_cell_manager(cm)
        self.pm.setup_load_balancer()

    @only_root
    def init_arrays(self):
        x = np.linspace(-1, 1, num=self.n, dtype=np.float32)
        y = np.zeros_like(x)

        indices = np.where(x ** 2 + y ** 2 > 0.1)
        x = x[indices]
        y = y[indices]

        vy = x.copy()
        vx = -1 * y.copy()
        self.x, self.y, self.vx, self.vy = wrap(x, y, vx, vy,
                                                backend=self.backend)
        self.colors = wrap(np.zeros(x.size, dtype=np.int32),
                           backend=self.backend)

    def step(self, x, y, vx, vy):
        self.step_func(x, y, vx, vy, self.dt)

    def set_colors(self, colors):
        colors[:] = self.pm.rank

    def solve(self, niter):
        for i in range(niter):
            dbg_print("Iteration no. %s" % (i + 1))
            dbg_print("%s %s" % (self.oe.x, self.oe.y))
            self.pm.update(self.oe.x, self.oe.y, migrate=True)
            #dbg_print(self.oe.vx)
            #self.step(self.oe.x, self.oe.y, self.oe.vx, self.oe.vy)
            #dbg_print(self.oe.vx)
            self.set_colors(self.oe.colors)
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
        cmap = np.array(['r', 'b'])
        plt.scatter(x, y, c=cmap[self.oe.colors])
        plt.show()


def run():
    solver = Solver(300, 0.01)
    solver.solve(3)
    solver.plot()
    solver.ctx.pop()


if __name__ == "__main__":
    run()
