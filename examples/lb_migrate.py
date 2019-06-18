from compyle.api import annotate, Elementwise, wrap, get_config, declare
from pyzoltan.hetero.load_balancer import adapt_lb, LoadBalancer
from pyzoltan.hetero.parallel_manager import only_root, ParallelManager
from pyzoltan.hetero.rcb import dbg_print, root_print
from pyzoltan.hetero.profile import profile
import numpy as np


@annotate
def step_euler(i, x, y, vx, vy, dt):
    x[i] += vx[i] * dt
    y[i] += vy[i] * dt

    vx[i] = -y[i]
    vy[i] = x[i]


class Solver(ParallelManager):
    def __init__(self, n, dt, lbfreq=1000, animate=False):
        super(Solver, self).__init__()
        self.step_func = Elementwise(step_euler, backend=self.backend)
        self.n = n
        self.dt = dt
        self.lbfreq = lbfreq
        self.x, self.y = None, None
        self.vx, self.vy = None, None
        self.colors = None
        self.animate = animate
        self.init_arrays()
        self.init_lb(self.lbfreq)

    def init_lb(self, lbfreq):
        self.lb = LoadBalancer(2, np.float32, backend=self.backend)
        self.lb.set_coords(x=self.x, y=self.y)
        self.lb.set_data(vx=self.vx, vy=self.vy, colors=self.colors)
        self.lb.set_lbfreq(lbfreq)
        self.lb.set_padding(1.)

    @only_root
    def init_arrays(self):
        x = np.linspace(-1, 1, num=self.n, dtype=np.float32)
        y = np.zeros_like(x)
        vy = x.copy()
        vx = -1 * y.copy()
        self.x, self.y, self.vx, self.vy = wrap(x, y, vx, vy,
                                                backend=self.backend)
        self.colors = wrap(np.zeros(x.size, dtype=np.int32),
                           backend=self.backend)

    #@adapt_lb(lb_name='lb')
    def step(self, x, y, vx, vy):
        self.step_func(x, y, vx, vy, self.dt)

    def set_colors(self, colors):
        colors[:] = self.lb.lb_obj.rank

    def solve(self, niter):
        for i in range(niter):
            dbg_print("Iteration no. %s" % (i + 1))
            self.lb.update(migrate=True)
            dbg_print("Min = %s, Max = %s" % (self.lb.lb_obj.min, self.lb.lb_obj.max))
            self.step(self.lb.lb_data.x, self.lb.lb_data.y,
                      self.lb.lb_data.vx, self.lb.lb_data.vy)
            self.set_colors(self.lb.lb_data.colors)
            if self.animate:
                self.lb.gather()
                self.animate_plot()
        self.lb.gather()

    @only_root
    def animate_plot(self):
        pass

    @only_root
    def plot(self):
        import matplotlib.pyplot as plt
        x = self.lb.lb_data.x
        y = self.lb.lb_data.y
        cmap = np.array(['r', 'b'])
        plt.scatter(x, y, c=cmap[self.lb.lb_data.colors])
        plt.show()

    @only_root
    def check(self):
        x = self.lb.lb_data.x.get()
        y = self.lb.lb_data.y.get()
        vx = self.lb.lb_data.vx.get()
        vy = self.lb.lb_data.vy.get()


#@profile(filename="profile_out")
def run():
    solver = Solver(300, 0.01)
    #solver.plot()
    solver.solve(160)
    solver.plot()
    #solver.check()


if __name__ == "__main__":
    run()
