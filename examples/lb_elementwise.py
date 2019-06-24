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


@annotate
def f(i, x, y, z, niter):
    for j in range(niter):
        z[i] = x[i] * x[i] + y[i] * y[i]


class MyObjectExchange(ObjectExchange):
    def __init__(self, x, y, z, backend):
        self.x = x
        self.y = y
        self.z = z
        self.backend = backend

    def transfer(self):
        x_new = carr.empty(self.plan.nreturn, np.float32,
                           backend=self.backend)
        y_new = carr.empty(self.plan.nreturn, np.float32,
                           backend=self.backend)
        z_new = carr.empty(self.plan.nreturn, np.float32,
                           backend=self.backend)

        self.plan.comm_do_post(self.x, x_new)
        self.plan.comm_do_post(self.y, y_new)
        self.plan.comm_do_post(self.z, z_new)

        self.plan.comm_do_wait()

        self.x = x_new
        self.y = y_new
        self.z = z_new

        return self.x, self.y


class Solver(object):
    def __init__(self, n, lbfreq=10):
        self.backend = 'cuda'
        self.ctx = make_context()
        self.func = Elementwise(f, backend=self.backend)
        self.n = n
        self.lbfreq = lbfreq
        self.x, self.y, self.z = None, None, None
        self.init_arrays()
        self.init_partition_manager()

    def init_partition_manager(self):
        self.pm = PartitionManager(2, np.float32, migrate=False,
                                   backend=self.backend)
        self.oe = MyObjectExchange(self.x, self.y, self.z, self.backend)
        cm = CellManager(2, np.float32, 0.1, backend=self.backend)
        self.pm.set_lbfreq(self.lbfreq)
        self.pm.set_object_exchange(self.oe)
        self.pm.set_cell_manager(cm)
        self.pm.setup_load_balancer()

    @only_root
    def init_arrays(self):
        x, y = np.mgrid[0:1:self.n*1j, 0:1:self.n*1j]
        x, y = x.ravel(), y.ravel()
        z = np.zeros_like(x)
        x, y, z = x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)
        self.x, self.y, self.z = wrap(x, y, z, backend=self.backend)

    def step(self, x, y, z):
        self.func(x, y, z, 65536)

    def solve(self):
        self.pm.update(self.oe.x, self.oe.y)
        self.step(self.oe.x, self.oe.y, self.oe.z)
        self.oe.gather()

    @only_root
    def check(self):
        x = self.oe.x.get()
        y = self.oe.y.get()
        z = self.oe.z.get()
        z_calc = x**2 + y**2
        assert np.allclose(z, z_calc)


#@profile(filename="profile_out")
def run():
    solver = Solver(5000)
    solver.solve()
    #solver.check()
    solver.ctx.pop()


if __name__ == "__main__":
    run()
