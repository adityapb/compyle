from compyle.api import annotate, Elementwise, wrap, get_config, declare
from pyzoltan.hetero.partition_manager import PartitionManager
from pyzoltan.hetero.cell_manager import CellManager
from pyzoltan.hetero.object_exchange import ObjectExchange

from pyzoltan.hetero.parallel_manager import only_root, ParallelManager
from pyzoltan.hetero.rcb import dbg_print, root_print
from pyzoltan.hetero.profile import profile
import numpy as np


@annotate
def f(i, x, y, z, niter):
    z[i] = 0
    for j in range(niter):
        z[i] += x[i] * x[i] + y[i] * y[i]


class Solver(object):
    def __init__(self, n, lbfreq=10):
        #super(Solver, self).__init__()
        self.backend = backend
        self.func = Elementwise(f, backend=self.backend)
        self.n = n
        self.lbfreq = lbfreq
        self.x, self.y, self.z = None, None, None
        self.init_arrays()
        self.init_partition_manager()

    def init_partition_manager(self):
        self.pm = PartitionManager(2, np.float32, backend=self.backend)
        obj_exchg = ObjectExchange()
        cm = CellManager(2, np.float32, 0.1, num_objs=self.n)
        self.pm.set_lbfreq(self.lbfreq)
        self.pm.set_object_exchange(obj_exchg)
        self.pm.set_cell_manager(cm)
        self.pm.set_coords([self.x, self.y])
        self.pm.setup_load_balancer()

    @only_root
    def init_arrays(self):
        x, y = np.mgrid[0:1:self.n*1j, 0:1:self.n*1j]
        x, y = x.ravel(), y.ravel()
        z = np.zeros_like(x)
        x, y, z = x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)
        self.x, self.y, self.z = wrap(x, y, z, backend=self.backend)

    #@adapt_lb(lb_name='lb')
    def step(self, x, y, z, niter):
        self.func(x, y, z, niter)

    def solve(self, niter):
        self.pm.update()

    @only_root
    def check(self):
        x = self.lb.lb_data.x.get()#.reshape((self.n, self.n))
        y = self.lb.lb_data.y.get()#.reshape((self.n, self.n))
        z = self.lb.lb_data.z.get()#.reshape((self.n, self.n))
        z_calc = x**2 + y**2
        assert np.allclose(z, z_calc)


@profile(filename="profile_out")
def run():
    solver = Solver(300)
    solver.solve(500)
    #solver.check()


if __name__ == "__main__":
    run()
