import numpy as np
import sys
sys.path.append('../../src')
from timeSchemes import *
import os
from dgCore import *
from eulerEquations import eulerEquations,vortexICS

if __name__== "__main__":
  L1 = 10.
  L2 = 10.
  Nelx = 2**5
  Nely = 2**5
  x = np.linspace(-L1/2.,L1/2.,Nelx+1)
  y = np.linspace(-L2/2.,L2/2.,Nely+1)
  x,y = np.meshgrid(x,y,indexing='ij')
  h = float(10./Nelx)
  x = x.flatten()
  y = y.flatten()
  X = np.zeros((2,(Nelx+1)*(Nely+1)))
  X[0] = x
  X[1] = y
  p = 3 #polynomial order
  quad_order = 6 #quadrature order
  grid = createGrid(X,p,quad_order,periodic=True)
  
  ## Initialize equation set
  eqns = eulerEquations()
  tri = grid.tri
  ### Initialize variables
  nvars = eqns.nvars
  U = np.zeros((nvars,grid.order_glob,grid.tri.nsimplex))
  U,xGlob = constructUIC_ell2Projection(grid,vortexICS)
  
  t = 0
  et = 10.
  dt = 0.005
  gamma = 1.4
  
  iteration = 0
  save_freq = 100
  t1 = time.time()
  if not os.path.exists('Solution'):
    os.makedirs('Solution')

  while (t <= et - 0.*dt/2):
    print('t = ' + str(t) + '  wall time = ' + str(time.time() -t1),'  Solution norm = ' + str( np.linalg.norm(U)))
    if (iteration%save_freq == 0):
      string = 'Solution/npsol' + str(iteration)
      eqns.writeSol(string,U,grid)
  
    iteration += 1
    U = advanceSolRK4(U,grid,eqns,dt)
    t += dt

  print('Finished!')
