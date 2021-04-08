import numpy as np
from dgCore import *
from dgHyperCore import *
class CreatePressioAdapter():
  def __init__(self,grid,eqns,hyper=False):
    self.grid = grid
    self.eqns = eqns
    self.N = eqns.nvars*grid.order_glob*grid.tri.nsimplex
    self.yShape = (eqns.nvars,grid.order_glob,grid.tri.nsimplex)

  def velocity(self,y,t,f):
    ytmp = np.reshape(y,self.yShape)
    ftmp = computeRHS(ytmp,self.grid.tri,self.eqns)
    f[:] = np.einsum('ijk,njk->nik',self.grid.Minv,ftmp).flatten()
    return f.flatten()

  def createVelocity(self):
    return np.zeros(self.N)

  def applyJacobian(self,y,B,t,A):
    f0 = self.createVelocity()
    self.velocity(y,t,f0)
    f1 = self.createVelocity()
    eps = 1e-4
    for i in range(0,np.shape(A)[1]):  
      ytmp = y + eps*B[:,i]
      self.velocity(ytmp,t,f1)
      A[:,i] = (f1 - f0)/eps 


  def createApplyJacobianResult(self, B):
    return np.zeros_like(B)


class CreatePressioHyperAdapter():
  def __init__(self,grid,eqns,hyper=False):
    self.grid = grid
    self.eqns = eqns
    self.N = eqns.nvars*grid.order_glob*grid.tri.Nelements
    self.yShape = (eqns.nvars,grid.order_glob,grid.tri.nsimplex)

  def velocity(self,y,t,f):
    ytmp = np.reshape(y,self.yShape)
    ftmp = computeRHSHyper(ytmp,self.grid.tri,self.eqns)
    f[:] = np.einsum('ijk,njk->nik',self.grid.Minv,ftmp).flatten()
    return f.flatten()

  def createVelocity(self):
    return np.zeros(self.N)

  def applyJacobian(self,y,B,t,A):
    f0 = self.createVelocity()
    self.velocity(y,t,f0)
    f1 = self.createVelocity()
    eps = 1e-4
    for i in range(0,np.shape(A)[1]):  
      ytmp = y + eps*B[:,i]
      self.velocity(ytmp,t,f1)
      A[:,i] = (f1 - f0)/eps 


  def createApplyJacobianResult(self, B):
    return np.zeros_like(B[self.grid.sampleElementsIdsForPressio])
