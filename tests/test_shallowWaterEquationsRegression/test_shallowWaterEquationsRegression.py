import sys
sys.path.append('src')
sys.path.append('../../src')
import numpy as np
from timeSchemes import *
import os
from dgCore import *
from shallowWaterEquations import * 


def boundaryConditions(grid,uInterior):
  uExterior = uInterior[:,:]*1.
  uExterior[1::] *= -1
  return uExterior

def test_main():
  L1 = 10.
  L2 = 10.
  Nelx = 2**3
  Nely = 2**3
  x = np.linspace(-L1/2.,L1/2.,Nelx+1)
  y = np.linspace(-L2/2.,L2/2.,Nely+1)
  x,y = np.meshgrid(x,y,indexing='ij')
  h = float(10./Nelx)
  x = x.flatten()
  y = y.flatten()
  X = np.zeros((2,(Nelx+1)*(Nely+1)))
  X[0] = x
  X[1] = y
  p = 2 #polynomial order
  quad_order = 4 #quadrature order
  grid = createGrid(X,p,quad_order)
  
  ## Initialize equation set
  eqns = shallowWaterEquations("CUSTOM_BCS",boundaryConditions)
  tri = grid.tri
  ### Initialize variables
  nvars = eqns.nvars
  U = np.zeros((nvars,grid.order_glob,grid.tri.nsimplex))
  U,xGlob = constructUIC_ell2Projection(grid,gaussianICS)
  
  t = 0
  et = 5.
  dt = 0.025
  
  iteration = 0
  save_freq = 5
  t1 = time.time()

  while (t <= et - dt/2):
    iteration += 1
    U = advanceSolRK4(U,grid,eqns,dt)
    t += dt

  data = np.load('tests/test_shallowWaterEquationsRegression/sol.gold.npz')
  assert np.allclose(U,data['U'])
