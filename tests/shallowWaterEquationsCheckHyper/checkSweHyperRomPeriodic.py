import sys
sys.path.append('../../src')
import numpy as np
from timeSchemes import *
import os
from dgHyperCore import *
from dgCore import *
from shallowWaterEquations import * 
# need to update for ordering
if __name__== "__main__":
  ## Create grid
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
  eqns = shallowWaterEquations()

  # create sample mesh
  N = grid.tri.nsimplex
  cells = np.array(range(0,N),dtype='int')
  cells = np.random.choice(cells,int(N*0.25),replace=False)
  hyperGrid = createHyperGrid(grid,eqns,cells)
  tri = grid.tri

  ### Initialize variables
  nvars = eqns.nvars
  U = np.zeros((nvars,grid.order_glob,grid.tri.nsimplex))
  U,xGlob = constructUIC_ell2Projection(grid,gaussianICS)
  
  t = 0
  dt = 0.005
  
  rhs1 = computeRHSHyper(U[:,:,hyperGrid.stencilElements],hyperGrid.tri,eqns)
  rhs2 = computeRHS(U,grid.tri,eqns)[:,:,hyperGrid.sampleElements]
  check = np.allclose(rhs1,rhs2)
  if check:
    print('Hyperreduction check passed')
  else:
    print("Oh dear...something failed. I shouldn't have been watching golf while I wrote this")

