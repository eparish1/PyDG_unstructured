import numpy as np
from dgCore import computeRHS
from scipy.sparse.linalg import LinearOperator
import scipy.optimize
def advanceSolRK4(U,grid,eqns,dt):
  U0 = np.zeros(np.shape(U))
  U0[:] = U[:]
  rk4const = np.array([1./4,1./3,1./2,1.])
  for i in range(0,4):
    RHS =  computeRHS(U,grid.tri,eqns)
    RHSsolve = np.einsum('ijk,njk->nik',grid.Minv,RHS)
    U[:] = U0[:] + dt*rk4const[i]*RHSsolve[:]
  return U



def SteadyStateBICGSTAB(U,grid,eqns,dt):
  N = np.size(U)
  def residual(v):
    v = np.reshape(v,np.shape(U))
    RHS = computeRHS(v,grid.tri,eqns)
    RHSsolve = np.einsum('ijk,njk->nik',grid.Minv,RHS)
    return RHSsolve

  solution = scipy.optimize.newton_krylov(residual,U.flatten(),method='bicgstab',verbose=3)
  solution = np.reshape(solution,np.shape(U))
  return solution
