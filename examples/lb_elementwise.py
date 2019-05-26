from compyle.api import annotate, Elementwise, wrap, get_config, declare
from pyzoltan.hetero.load_balancer import adapt_lb, LoadBalancer
from pyzoltan.hetero.parallel_manager import only_root, ParallelManager
from pyzoltan.hetero.rcb import dbg_print, root_print
from pyzoltan.hetero.profile import profile
import numpy as np


@annotate
def f(i, x, y, z, niter):
    z[i] = 0
    for j in range(niter):
        z[i] += x[i] * x[i] + y[i] * y[i]


class Solver(ParallelManager):
    def __init__(self, n, lbfreq=10):
        super(Solver, self).__init__()
        self.func = Elementwise(f, backend=self.backend)
        self.n = n
        self.lbfreq = lbfreq
        self.x, self.y, self.z = None, None, None
        self.init_arrays()
        self.init_lb(self.lbfreq)

    def init_lb(self, lbfreq):
        self.lbfreq = lbfreq
        self.lb = LoadBalancer(2, np.float32, backend=self.backend)
        self.lb.set_coords(x=self.x, y=self.y)
        self.lb.set_data(z=self.z)

    @only_root
    def init_arrays(self):
        x, y = np.mgrid[0:1:self.n*1j, 0:1:self.n*1j]
        x, y = x.ravel(), y.ravel()
        z = np.zeros_like(x)
        x, y, z = x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)
        self.x, self.y, self.z = wrap(x, y, z, backend=self.backend)

    @adapt_lb(lb_name='lb')
    def step(self, x, y, z, niter):
        self.func(x, y, z, niter)

    def solve(self, niter):
        for i in range(niter):
            dbg_print("Iteration no. %s" % (i + 1))
            if i % self.lbfreq == 0:
                self.lb.load_balance()
                dbg_print("Weight = %s" % self.lb.lb_data.proc_weights)
            self.step(self.lb.lb_data.x, self.lb.lb_data.y, self.lb.lb_data.z,
                      1000)
        self.lb.gather()

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
