import sys
sys.path.append('../../src')
import numpy as np
from timeSchemes import *
import os
from dgHyperCore import *
from dgCore import *
from shallowWaterEquations import * 

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
  grid = createGrid(X,p,quad_order)
  N = grid.tri.nsimplex
  cells = np.array(range(0,N),dtype='int')
  cells = np.random.choice(cells,int(N*0.1),replace=False)
  hyperGrid = createHyperGrid(grid,cells)
  ## Initialize equation set
  eqns = shallowWaterEquations()
  tri = grid.tri
  ### Initialize variables
  nvars = eqns.nvars
  U = np.zeros((nvars,grid.order_glob,grid.tri.nsimplex))
  U,xGlob = constructUIC_ell2Projection(grid,gaussianICS)
  
  t = 0
  et = 10.
  dt = 0.005
  gamma = 1.4
  
  iteration = 0
  save_freq = 10
  t1 = time.time()
  if not os.path.exists('Solution'):
    os.makedirs('Solution')

  while (t <= et - 0.*dt/2):
    print('t = ' + str(t) + '  wall time = ' + str(time.time() -t1),'  Solution norm = ' + str( np.linalg.norm(U)))
    if (iteration%save_freq == 0):
      string = 'Solution/npsol' + str(iteration)
      eqns.writeSol(string,U,grid)
  
    test = computeRHSHyper(U[:,:,hyperGrid.stencilElements],hyperGrid.tri,eqns)[:,:,hyperGrid.sampleElementsIds]
    test2 = computeRHS(U,grid.tri,eqns)[:,:,hyperGrid.sampleElements]
    check = np.linalg.norm( test - test2)
    print('Hyper norm = ' + str(check))
    iteration += 1
    U = advanceSolRK4(U,grid,eqns,dt)
    t += dt

  print('Finished!')
