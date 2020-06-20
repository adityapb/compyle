import hoomd
import sys
from hoomd import md

hoomd.context.initialize("")
ndim = int(sys.argv[1])

# Create a 10x10x10 simple cubic lattice of particles with type name A
data = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0, type_name='A'), n=[ndim, ndim])

#snapshot = data.take_snapshot()

#print(snapshot.particles.position)

# Specify Lennard-Jones interactions between particle pairs
nl = md.nlist.cell(r_buff=0)
lj = md.pair.lj(r_cut=3.0, nlist=nl)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)

# Integrate at constant temperature
md.integrate.mode_standard(dt=0.02)
hoomd.md.integrate.nve(group=hoomd.group.all())

# Run for 10,000 time steps
hoomd.run(100)
