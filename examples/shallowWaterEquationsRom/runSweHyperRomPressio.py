import sys
sys.path.append('../../src')
import numpy as np
from timeSchemes import *
import os
from dgCore import *
from dgHyperCore import *
from shallowWaterEquations import * 
from pressio4PyAdapter import *
# if run from within a build of pressio4py, need to append to python path
import pathlib, sys
file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(file_path) + "/..")         # to access doFom
from pressio4py import rom as rom, logger
from pressio4py import solvers as solvers
from scipy import linalg
  # this linear solver is used at each gauss-newton iteration
class MyLinSolver:
  def __init__(self): pass
  def solve(self, A,b,x):
    lumat, piv, info = linalg.lapack.dgetrf(A, overwrite_a=True)
    x[:], info = linalg.lapack.dgetrs(lumat, piv, b, 0, 0)

def boundaryConditions(grid,uInterior):
  uExterior = uInterior[:,:]*1.
  uExterior[1::] *= -1
  return uExterior


class RomStateObserver:
  def __init__(self): pass
  def __call__(self, timeStep, time, state):
    print('time = ' + str(time))

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
  ## Initialize equation set
  eqns = shallowWaterEquations("CUSTOM_BCS",boundaryConditions)

  N = grid.tri.nsimplex
  cells = np.array(range(0,N),dtype='int')
  cells = np.random.choice(cells,int(N*0.1),replace=False)
  hyperGrid = createHyperGrid(grid,eqns,cells)

  ## create adapter
  fomObj = CreatePressioHyperAdapter(hyperGrid,eqns,hyper=False)
  Phi = np.load('pod_basis.npz')['Phi']
  Phi = np.reshape(Phi,(eqns.nvars,grid.order_glob,grid.tri.nsimplex,np.shape(Phi)[1]))
  Phi = Phi[:,:,hyperGrid.stencilElements]
  Phi = np.reshape(Phi,(eqns.nvars*grid.order_glob*np.size(hyperGrid.stencilElements),np.shape(Phi)[-1]))
  Phi = np.reshape(Phi.flatten(order='F'),np.shape(Phi),order='F')
  


  romSize = np.shape(Phi)[1]
  linearDecoder = rom.Decoder(Phi)

  fomReferenceState = np.zeros(np.shape(Phi)[0])
  
  tri = grid.tri
  ### Initialize variables
  U = np.zeros((eqns.nvars,grid.order_glob,grid.tri.nsimplex))
  U,xGlob = constructUIC_ell2Projection(grid,gaussianICS)

  fomInitialState = copy.deepcopy(U[:,:,hyperGrid.stencilElements])
  romState = np.dot(Phi.transpose(),fomInitialState.flatten())
  problem = rom.lspg.unsteady.hyperreduced.ProblemEuler(fomObj, linearDecoder, romState, fomReferenceState,hyperGrid.sampleElementsIdsForPressio)  
  t = 0
  et = 10.
  dt = 0.005
  nsteps = int(et/dt) 

  # create the Gauss-Newton solver
  nonLinSolver = solvers.createGaussNewton(problem, romState, MyLinSolver())
  # set tolerance and convergence criteria
  nlsTol, nlsMaxIt = 1e-6, 5
  nonLinSolver.setMaxIterations(nlsMaxIt)
  nonLinSolver.setStoppingCriterion(solvers.stop.whenCorrectionAbsoluteNormBelowTolerance)

  # create object to monitor the romState at every iteration
  myObs = RomStateObserver()
  # solve problem
  rom.lspg.solveNSequentialMinimizations(problem, romState, 0., dt, nsteps, myObs,nonLinSolver)

  print('Finished!')
